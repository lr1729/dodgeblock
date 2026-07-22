import { ARENA_X, BLOCK_H, SPAWN_GRID } from '../src/constants.js';
import { Rng } from '../src/sim/rng.js';
import { Sim } from '../src/sim/sim.js';
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

const count = numberArg('--envs', 64);
const baseSeed = numberArg('--seed', 1) >>> 0;
const archiveProbability = Math.max(0, Math.min(1, numberArg('--archive-probability', 0)));
const archiveCapacity = Math.max(1, numberArg('--archive-capacity', 2048));
const checkpointHeight = Math.max(BLOCK_H, numberArg('--archive-height', 100));
const rules = Object.freeze({ autoGuard: false, checkpoints: false });
const observationBytes = observationByteSize();
const statsBytes = count * (4 + 1 + 4 + 4 + 4 + 4 + 4 + 4 + 4);
const packetBytes = count * observationBytes + statsBytes;
const archive = [];
const archiveFingerprints = new Set();
const archiveByBucket = new Map();
const archiveRng = new Rng(baseSeed ^ 0xa5a5a5a5);
let archiveCursor = 0;

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
  saved.starts++;
  return true;
}

function reset(entry, index) {
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
  if (!sampleArchive(entry, index)) entry.sim = freshSim(index, entry.episode);
}

function copyFloatArray(packet, values, offset) {
  Buffer.from(values.buffer, values.byteOffset, values.byteLength).copy(packet, offset);
  return offset + values.byteLength;
}

function writePacket(
  rewards = null,
  dones = null,
  returns = null,
  lengths = null,
  heights = null,
  episodeStarts = null,
  worldScales = null,
) {
  const packet = Buffer.allocUnsafe(packetBytes);
  const terrainOffset = 0;
  const skylineOffset = terrainOffset + count * TERRAIN_SIZE;
  const fallingOffset = skylineOffset + count * SKYLINE_SIZE;
  const fallingBytes = FALLING_COUNT * FALLING_FEATURES * 4;
  const forecastOffset = fallingOffset + count * fallingBytes;
  const forecastBytes = FORECAST_COUNT * FORECAST_FEATURES * 4;
  const stateOffset = forecastOffset + count * forecastBytes;
  const stateBytes = STATE_SIZE * 4;
  for (let index = 0; index < envs.length; index++) {
    const entry = envs[index];
    const observation = encodeObservation(entry.sim, entry, entry.observation);
    Buffer.from(observation.terrain.buffer).copy(packet, terrainOffset + index * TERRAIN_SIZE);
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

function step(actions) {
  const rewards = new Float32Array(count);
  const dones = new Uint8Array(count);
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
    worldScales[index] = transition.worldScale;
    entry.previousAction = action;
    const reward = Math.max(0, entry.sim.height - beforeHeight) / BLOCK_H;
    entry.episodeReturn += reward;
    entry.episodeLength++;
    rewards[index] = reward;
    archiveState(entry);
    if (!entry.sim.dead) continue;
    dones[index] = 1;
    returns[index] = entry.episodeReturn;
    lengths[index] = entry.episodeLength;
    heights[index] = entry.sim.height;
    episodeStarts[index] = entry.startHeight;
    reset(entry, index);
  }
  writePacket(rewards, dones, returns, lengths, heights, episodeStarts, worldScales);
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
