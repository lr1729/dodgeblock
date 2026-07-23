import fs from 'node:fs';
import zlib from 'node:zlib';

import { ARENA_X, BLOCK_H, SPAWN_GRID } from '../src/constants.js';
import { Rng } from '../src/sim/rng.js';
import { Sim } from '../src/sim/sim.js';
import { heightReward, targetReward } from './reward-v2.mjs';
import {
  ACTION_COUNT,
  FALLING_COUNT,
  FALLING_FEATURES,
  FORECAST_COUNT,
  FORECAST_FEATURES,
  SKYLINE_SIZE,
  STATE_SIZE,
  TERRAIN_SIZE,
  createObservation,
  encodeObservation,
  heldActionInput,
  observationByteSize,
} from './env-v2.mjs';

function numberArg(name, fallback) {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : Number(process.argv[index + 1]);
}

function stringArg(name, fallback) {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : process.argv[index + 1];
}

function repeatedArgs(name) {
  const values = [];
  for (let index = 0; index < process.argv.length - 1; index++) {
    if (process.argv[index] === name) values.push(process.argv[index + 1]);
  }
  return values;
}

const count = numberArg('--envs', 64);
const baseSeed = numberArg('--seed', 1) >>> 0;
const archiveProbability = Math.max(0, Math.min(1, numberArg('--archive-probability', 0)));
const archiveCapacity = Math.max(1, numberArg('--archive-capacity', 2048));
const checkpointHeight = Math.max(BLOCK_H, numberArg('--archive-height', 100));
const deathPenalty = Math.max(0, numberArg('--death-penalty', 1));
const aliveReward = Math.max(0, numberArg('--alive-reward', 0));
const targetHeight = Math.max(BLOCK_H, numberArg('--target-height', 10_000));
const discount = Math.max(0, Math.min(1, numberArg('--discount', 0.99999)));
const rewardMode = stringArg('--reward-mode', 'height');
if (!['height', 'target'].includes(rewardMode)) {
  throw new Error(`unsupported reward mode: ${rewardMode}`);
}
const rules = Object.freeze({ autoGuard: false, checkpoints: false });
const demonstrationPaths = repeatedArgs('--demonstration');
const demonstrationProbabilityStart = Math.max(
  0,
  Math.min(1, numberArg('--demonstration-probability', demonstrationPaths.length ? 0.5 : 0)),
);
const demonstrationProbabilityEnd = Math.max(
  0,
  Math.min(
    demonstrationProbabilityStart,
    numberArg('--demonstration-probability-end', demonstrationPaths.length ? 0.2 : 0),
  ),
);
const demonstrationSnapshotInterval = Math.max(
  1,
  Math.floor(numberArg('--demonstration-snapshot-interval', 30)),
);
const demonstrationSnapshotCapacity = Math.max(
  1,
  Math.floor(numberArg('--demonstration-snapshot-capacity', 256)),
);
const reverseCurriculumInitialFrames = Math.max(
  1,
  Math.floor(numberArg('--reverse-curriculum-initial-frames', 60)),
);
const demonstrationRandomizeProbability = Math.max(
  0,
  Math.min(1, numberArg('--demonstration-randomize-probability', 1)),
);
const observationBytes = observationByteSize();
const statsBytes = count * (4 + 1 + 1 + 4 + 4 + 4 + 4 + 4 + 4 + 4);
const packetBytes = count * observationBytes + statsBytes;
const archive = [];
const archiveFingerprints = new Set();
const archiveByBucket = new Map();
const archiveRng = new Rng(baseSeed ^ 0xa5a5a5a5);
let archiveCursor = 0;

function loadDemonstration(filename) {
  const payload = JSON.parse(zlib.gunzipSync(fs.readFileSync(filename)));
  if (payload.version !== 1) {
    throw new Error(`unsupported demonstration version in ${filename}: ${payload.version}`);
  }
  const actions = Buffer.from(payload.actions, 'base64');
  if (actions.length !== payload.frames) {
    throw new Error(`action length mismatch in ${filename}`);
  }
  const sim = new Sim(payload.seed, { rules });
  const denseTailCapacity = Math.min(
    64,
    Math.max(1, Math.floor(demonstrationSnapshotCapacity / 4)),
  );
  const globalSnapshotCapacity = Math.max(
    1,
    demonstrationSnapshotCapacity - denseTailCapacity - 1,
  );
  const effectiveSnapshotInterval = Math.max(
    demonstrationSnapshotInterval,
    Math.ceil(actions.length / globalSnapshotCapacity),
  );
  const denseTailInterval = Math.max(1, Math.floor(reverseCurriculumInitialFrames / 4));
  const captureFrames = new Set([
    0,
    Math.max(0, actions.length - reverseCurriculumInitialFrames),
    Math.max(0, actions.length - 1),
  ]);
  for (
    let frame = effectiveSnapshotInterval;
    frame < actions.length;
    frame += effectiveSnapshotInterval
  ) {
    captureFrames.add(frame);
  }
  for (
    let remaining = denseTailInterval;
    remaining <= denseTailInterval * denseTailCapacity && remaining < actions.length;
    remaining += denseTailInterval
  ) {
    captureFrames.add(actions.length - remaining);
  }
  const snapshots = [{
    frame: 0,
    remaining: actions.length,
    previousAction: 0,
    snapshot: sim.snapshot(),
  }];
  let previousAction = 0;
  for (let frame = 0; frame < actions.length; frame++) {
    const action = actions[frame];
    sim.step(heldActionInput(action, previousAction));
    previousAction = action;
    if (sim.dead) throw new Error(`demonstration ${filename} dies at frame ${sim.frame}`);
    if (captureFrames.has(frame + 1)) {
      snapshots.push({
        frame: frame + 1,
        remaining: actions.length - frame - 1,
        previousAction,
        snapshot: sim.snapshot(),
      });
    }
  }
  if (sim.height < targetHeight) {
    throw new Error(`demonstration ${filename} reaches ${sim.height}, below target ${targetHeight}`);
  }
  if (payload.finalHash && sim.hash() !== payload.finalHash) {
    throw new Error(`demonstration ${filename} has a mismatched final state hash`);
  }
  return {
    filename,
    seed: payload.seed,
    frames: actions.length,
    snapshotInterval: effectiveSnapshotInterval,
    denseTailInterval,
    snapshots,
  };
}

const demonstrations = demonstrationPaths.map(loadDemonstration);
for (const demonstration of demonstrations) {
  process.stderr.write(`${JSON.stringify({
    event: 'demonstration_loaded',
    filename: demonstration.filename,
    frames: demonstration.frames,
    snapshots: demonstration.snapshots.length,
    snapshotInterval: demonstration.snapshotInterval,
    denseTailInterval: demonstration.denseTailInterval,
  })}\n`);
}
const longestDemonstration = demonstrations.reduce(
  (longest, demonstration) => Math.max(longest, demonstration.frames),
  reverseCurriculumInitialFrames,
);
let curriculumMaximumRemaining = Math.min(longestDemonstration, reverseCurriculumInitialFrames);
let curriculumFrontierAttempts = 0;
let curriculumFrontierSuccesses = 0;

function curriculumProgress() {
  if (longestDemonstration <= reverseCurriculumInitialFrames) return 1;
  return Math.max(0, Math.min(
    1,
    (curriculumMaximumRemaining - reverseCurriculumInitialFrames) /
      (longestDemonstration - reverseCurriculumInitialFrames),
  ));
}

function currentDemonstrationProbability() {
  const progress = curriculumProgress();
  return demonstrationProbabilityStart +
    progress * (demonstrationProbabilityEnd - demonstrationProbabilityStart);
}

function episodeSeed(index, episode) {
  return (baseSeed + index * 0x9e3779b9 + episode * 0x85ebca6b) >>> 0;
}

function freshSim(index, episode) {
  return new Sim(episodeSeed(index, episode), { rules });
}

const envs = Array.from({ length: count }, (_, index) => ({
  sim: freshSim(index, 0),
  observation: createObservation(),
  previousAction: 0,
  episodeReturn: 0,
  episodeLength: 0,
  episode: 0,
  startHeight: 0,
  nextArchiveHeight: checkpointHeight,
  archiveOrigin: null,
  curriculumSource: 'fresh',
  demonstrationRemaining: 0,
  demonstrationFrontier: false,
}));

function isStableArchiveState(sim) {
  const p = sim.player;
  return p.offGround === 0 &&
    p.focusAimTimer <= 0 &&
    p.focusTimer <= 0 &&
    (!p.supportBlock || (p.supportBlock.fixed && p.supportBlock.faultTimer <= 0));
}

function archiveFingerprint(sim, bucket) {
  const surface = sim.blocks.surfaceLayers();
  const relative = surface.map((layer) => Math.max(-8, Math.min(8, layer - Math.round(bucket / BLOCK_H))));
  const playerColumn = Math.round((sim.player.x - ARENA_X) / SPAWN_GRID);
  return `${bucket}|${playerColumn}|${sim.player.focus}|${relative.join(',')}`;
}

function removeArchiveEntry(entry) {
  archiveFingerprints.delete(entry.fingerprint);
  const bucketEntries = archiveByBucket.get(entry.bucket);
  if (bucketEntries) {
    const index = bucketEntries.indexOf(entry);
    if (index !== -1) bucketEntries.splice(index, 1);
    if (!bucketEntries.length) archiveByBucket.delete(entry.bucket);
  }
}

function archiveState(entry) {
  const sim = entry.sim;
  if (sim.height < entry.nextArchiveHeight || !isStableArchiveState(sim)) return;
  const bucket = Math.floor(sim.height / checkpointHeight) * checkpointHeight;
  entry.nextArchiveHeight = bucket + checkpointHeight;
  const fingerprint = archiveFingerprint(sim, bucket);
  if (archiveFingerprints.has(fingerprint)) return;
  const saved = {
    bucket,
    fingerprint,
    snapshot: sim.snapshot(),
    previousAction: entry.previousAction,
    starts: 0,
    advances: 0,
  };
  if (archive.length < archiveCapacity) {
    archive.push(saved);
  } else {
    const replaced = archive[archiveCursor];
    removeArchiveEntry(replaced);
    archive[archiveCursor] = saved;
    archiveCursor = (archiveCursor + 1) % archiveCapacity;
  }
  archiveFingerprints.add(fingerprint);
  if (!archiveByBucket.has(bucket)) archiveByBucket.set(bucket, []);
  archiveByBucket.get(bucket).push(saved);
}

function sampleArchive(entry, index) {
  if (!archive.length || !archiveRng.chance(archiveProbability)) return false;
  const buckets = [...archiveByBucket.keys()].sort((a, b) => a - b);
  const bucket = archiveRng.pick(buckets);
  const candidates = archiveByBucket.get(bucket);
  const saved = archiveRng.weighted(candidates.map((candidate) => {
    const advanceRate = (candidate.advances + 1) / (candidate.starts + 2);
    const learningPotential = 4 * advanceRate * (1 - advanceRate);
    const staleness = 1 / Math.sqrt(candidate.starts + 1);
    return { value: candidate, w: 0.25 + learningPotential + staleness };
  }));
  const sim = freshSim(index, entry.episode);
  sim.restore(saved.snapshot);
  // Forecasts and falling blocks were already visible at the archived state.
  // Resample only unrevealed future randomness so curriculum starts do not
  // leak a fixed future sequence into the learned policy.
  sim.seed = episodeSeed(index, entry.episode);
  sim.rng.s = sim.seed;
  sim.director.materialBag = [];
  entry.sim = sim;
  entry.previousAction = saved.previousAction;
  entry.startHeight = sim.height;
  entry.nextArchiveHeight = bucket + checkpointHeight;
  entry.archiveOrigin = saved;
  entry.curriculumSource = 'archive';
  saved.starts++;
  return true;
}

function sampleDemonstration(entry, index) {
  if (!demonstrations.length ||
    !archiveRng.chance(currentDemonstrationProbability())) return false;
  const demonstration = archiveRng.pick(demonstrations);
  const maximumRemaining = Math.min(demonstration.frames, curriculumMaximumRemaining);
  const candidates = demonstration.snapshots.filter(
    (candidate) => candidate.remaining > 0 && candidate.remaining <= maximumRemaining,
  );
  if (!candidates.length) {
    throw new Error(`demonstration ${demonstration.filename} has no curriculum snapshot`);
  }
  const saved = archiveRng.weighted(candidates.map((candidate) => ({
    value: candidate,
    w: 0.25 + candidate.remaining / Math.max(1, maximumRemaining),
  })));
  const sim = freshSim(index, entry.episode);
  sim.restore(saved.snapshot);
  const progress = maximumRemaining / demonstration.frames;
  if (archiveRng.chance(demonstrationRandomizeProbability * progress)) {
    sim.seed = episodeSeed(index, entry.episode);
    sim.rng.s = sim.seed;
    sim.director.materialBag = [];
  }
  entry.sim = sim;
  entry.previousAction = saved.previousAction;
  entry.startHeight = sim.height;
  entry.nextArchiveHeight = Math.floor(sim.height / checkpointHeight) * checkpointHeight +
    checkpointHeight;
  entry.archiveOrigin = null;
  entry.curriculumSource = 'demonstration';
  entry.demonstrationRemaining = saved.remaining;
  entry.demonstrationFrontier = saved.remaining >= maximumRemaining * 0.5;
  return true;
}

function updateReverseCurriculum(entry, success) {
  if (entry.curriculumSource !== 'demonstration' || !entry.demonstrationFrontier) return;
  curriculumFrontierAttempts++;
  if (success) curriculumFrontierSuccesses++;
  if (curriculumFrontierAttempts < 64) return;

  const successRate = curriculumFrontierSuccesses / curriculumFrontierAttempts;
  const previous = curriculumMaximumRemaining;
  if (successRate >= 0.6 && curriculumMaximumRemaining < longestDemonstration) {
    curriculumMaximumRemaining = Math.min(
      longestDemonstration,
      Math.max(
        curriculumMaximumRemaining + demonstrationSnapshotInterval,
        Math.round(curriculumMaximumRemaining * 1.35),
      ),
    );
  } else if (
    successRate < 0.15 &&
    curriculumMaximumRemaining > reverseCurriculumInitialFrames
  ) {
    curriculumMaximumRemaining = Math.max(
      reverseCurriculumInitialFrames,
      Math.round(curriculumMaximumRemaining * 0.9),
    );
  }
  process.stderr.write(`${JSON.stringify({
    event: 'reverse_curriculum',
    successRate,
    previousMaximumRemaining: previous,
    maximumRemaining: curriculumMaximumRemaining,
    longestDemonstration,
    progress: curriculumProgress(),
    demonstrationProbability: currentDemonstrationProbability(),
    randomizeProbability: demonstrationRandomizeProbability * curriculumProgress(),
  })}\n`);
  curriculumFrontierAttempts = 0;
  curriculumFrontierSuccesses = 0;
}

function reset(entry, index, success = false) {
  updateReverseCurriculum(entry, success);
  if (entry.archiveOrigin && entry.sim.height >= entry.startHeight + checkpointHeight) {
    entry.archiveOrigin.advances++;
  }
  entry.episode++;
  entry.previousAction = 0;
  entry.episodeReturn = 0;
  entry.episodeLength = 0;
  entry.startHeight = 0;
  entry.nextArchiveHeight = checkpointHeight;
  entry.archiveOrigin = null;
  entry.curriculumSource = 'fresh';
  entry.demonstrationRemaining = 0;
  entry.demonstrationFrontier = false;
  if (!sampleDemonstration(entry, index) && !sampleArchive(entry, index)) {
    entry.sim = freshSim(index, entry.episode);
  }
}

function copyFloatArray(packet, values, offset) {
  Buffer.from(values.buffer, values.byteOffset, values.byteLength).copy(packet, offset);
  return offset + values.byteLength;
}

function writePacket(
  rewards = null,
  dones = null,
  successes = null,
  returns = null,
  lengths = null,
  heights = null,
  episodeStarts = null,
  worldScales = null,
) {
  const packet = Buffer.allocUnsafe(packetBytes);
  const terrainOffset = 0;
  const skylineOffset = terrainOffset + count * TERRAIN_SIZE * 2;
  const fallingOffset = skylineOffset + count * SKYLINE_SIZE;
  const fallingBytes = FALLING_COUNT * FALLING_FEATURES * 4;
  const forecastOffset = fallingOffset + count * fallingBytes;
  const forecastBytes = FORECAST_COUNT * FORECAST_FEATURES * 4;
  const stateOffset = forecastOffset + count * forecastBytes;
  const stateBytes = STATE_SIZE * 4;
  for (let index = 0; index < envs.length; index++) {
    const entry = envs[index];
    const observation = encodeObservation(entry.sim, entry, entry.observation);
    Buffer.from(observation.terrain.buffer).copy(packet, terrainOffset + index * TERRAIN_SIZE * 2);
    Buffer.from(observation.skyline.buffer).copy(packet, skylineOffset + index * SKYLINE_SIZE);
    copyFloatArray(packet, observation.falling, fallingOffset + index * fallingBytes);
    copyFloatArray(packet, observation.forecasts, forecastOffset + index * forecastBytes);
    copyFloatArray(packet, observation.state, stateOffset + index * stateBytes);
  }
  let offset = stateOffset + count * stateBytes;
  for (let index = 0; index < count; index++) packet.writeFloatLE(rewards?.[index] ?? 0, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet[offset + index] = dones?.[index] ?? 0;
  offset += count;
  for (let index = 0; index < count; index++) packet[offset + index] = successes?.[index] ?? 0;
  offset += count;
  for (let index = 0; index < count; index++) packet.writeFloatLE(returns?.[index] ?? 0, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet.writeUInt32LE(lengths?.[index] ?? 0, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet.writeFloatLE(heights?.[index] ?? 0, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet.writeFloatLE(episodeStarts?.[index] ?? 0, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet.writeFloatLE(envs[index].startHeight, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet.writeFloatLE(envs[index].sim.height, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet.writeFloatLE(worldScales?.[index] ?? 1, offset + index * 4);
  process.stdout.write(packet);
}

function transitionReward(beforeHeight, sim, worldScale, success) {
  if (rewardMode === 'height') {
    return heightReward({
      beforeHeight,
      afterHeight: sim.height,
      blockHeight: BLOCK_H,
      dead: sim.dead,
      deathPenalty,
      aliveReward,
      worldScale,
    });
  }

  return targetReward({
    beforeHeight,
    afterHeight: sim.height,
    targetHeight,
    discount,
    worldScale,
    dead: sim.dead,
    success,
  });
}

function step(actions) {
  const rewards = new Float32Array(count);
  const dones = new Uint8Array(count);
  const successes = new Uint8Array(count);
  const returns = new Float32Array(count);
  const lengths = new Uint32Array(count);
  const heights = new Float32Array(count);
  const episodeStarts = new Float32Array(count);
  const worldScales = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const entry = envs[index];
    const beforeHeight = entry.sim.height;
    const action = Math.min(ACTION_COUNT - 1, actions[index]);
    const transition = entry.sim.step(heldActionInput(action, entry.previousAction));
    const success = !entry.sim.dead && entry.sim.height >= targetHeight;
    worldScales[index] = transition.worldScale;
    entry.previousAction = action;
    const reward = transitionReward(beforeHeight, entry.sim, transition.worldScale, success);
    entry.episodeReturn += reward;
    entry.episodeLength++;
    rewards[index] = reward;
    archiveState(entry);
    if (!entry.sim.dead && !success) continue;
    dones[index] = 1;
    successes[index] = success ? 1 : 0;
    returns[index] = entry.episodeReturn;
    lengths[index] = entry.episodeLength;
    heights[index] = entry.sim.height;
    episodeStarts[index] = entry.startHeight;
    reset(entry, index, success);
  }
  writePacket(rewards, dones, successes, returns, lengths, heights, episodeStarts, worldScales);
}

for (let index = 0; index < envs.length; index++) {
  sampleDemonstration(envs[index], index);
}
writePacket();
let pending = Buffer.alloc(0);
process.stdin.on('data', (chunk) => {
  pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
  while (pending.length >= count) {
    const actions = pending.subarray(0, count);
    pending = pending.subarray(count);
    step(actions);
  }
});

process.on('SIGTERM', () => process.exit(0));
