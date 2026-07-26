import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { BLOCK_H } from '../src/constants.js';
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
const deathPenalty = Math.max(0, numberArg('--death-penalty', 1));
const aliveReward = Math.max(0, numberArg('--alive-reward', 0));
const targetHeight = Math.max(BLOCK_H, numberArg('--target-height', 10_000));
const discount = Math.max(0, Math.min(1, numberArg('--discount', 0.99999)));
// Hold each chosen action for N sim frames. Tallec et al. (arXiv:1901.09732)
// show the action gap Q(s,a) - V(s) vanishes as the control interval shrinks,
// and this project measured exactly that fingerprint: held-out direction loss
// ~ ln(3), i.e. a policy whose action choice barely matters per frame. Repeating
// an action multiplies its effect and makes exploration temporally correlated --
// which is how the death autopsy's escapes were found (sticky random, mean hold
// ~7.5 frames). Applied here rather than in the trainer so one agent step is one
// transition and the rollout, GAE and loss are untouched.
const actionRepeat = Math.max(1, Math.round(numberArg('--action-repeat', 1)));
const rewardMode = stringArg('--reward-mode', 'height');
if (!['height', 'target'].includes(rewardMode)) {
  throw new Error(`unsupported reward mode: ${rewardMode}`);
}
const rules = Object.freeze({ autoGuard: false, checkpoints: false });
const cellBankPaths = repeatedArgs('--cell-bank');
const deathCaseDir = stringArg('--death-case-dir', '');
if (deathCaseDir) fs.mkdirSync(deathCaseDir, { recursive: true });
const observationBytes = observationByteSize();
const statsBytes = count * (4 + 1 + 1 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 1 + 1 + 1 + 1 + 1);
const packetBytes = count * observationBytes + statsBytes;
const DEATH_CAPTURE_STRIDE = 120;
const DEATH_CAPTURE_ANCHORS = 4;

function loadCellBank(filename) {
  const payload = JSON.parse(zlib.gunzipSync(fs.readFileSync(filename)));
  if (payload.version !== 1) {
    throw new Error(`unsupported cell bank version in ${filename}: ${payload.version}`);
  }
  if (Number(payload.targetHeight ?? targetHeight) !== targetHeight) {
    throw new Error(`cell bank target does not match ${targetHeight}: ${filename}`);
  }
  return payload.entries.map((entry) => ({
    key: entry.key,
    snapshot: entry.snapshot,
    previousAction: entry.previousAction,
  }));
}

const cellVariants = cellBankPaths.flatMap(loadCellBank);

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
  curriculumSource: 'fresh',
  cellVariantId: -1,
  captureAnchors: [],
  captureActions: [],
}));

function restoreCell(entry, index, variantId) {
  if (variantId < 0) {
    entry.sim = freshSim(index, entry.episode);
    entry.cellVariantId = -1;
    entry.curriculumSource = 'fresh';
    return;
  }
  const saved = cellVariants[variantId];
  if (!saved) throw new Error(`unknown cell variant ${variantId}`);
  const sim = freshSim(index, entry.episode);
  sim.restore(saved.snapshot);
  // Preserve every visible and causal state variable. Only hidden future RNG
  // and the ordering of the remaining material multiset are randomized.
  sim.seed = episodeSeed(index, entry.episode);
  sim.rng.s = sim.seed;
  sim.director.reshuffleMaterialRemainder();
  entry.sim = sim;
  entry.previousAction = saved.previousAction;
  entry.startHeight = sim.height;
  entry.cellVariantId = variantId;
  entry.curriculumSource = 'cell';
}

function reset(entry, index, variantId = -1) {
  entry.episode++;
  entry.previousAction = 0;
  entry.episodeReturn = 0;
  entry.episodeLength = 0;
  entry.startHeight = 0;
  entry.curriculumSource = 'fresh';
  entry.cellVariantId = -1;
  restoreCell(entry, index, variantId);
  resetDeathCapture(entry);
}

function resetDeathCapture(entry) {
  entry.captureActions = [];
  entry.captureAnchors = deathCaseDir
    ? [{
        episodeLength: 0,
        snapshot: entry.sim.snapshot(),
        previousAction: entry.previousAction,
      }]
    : [];
}

function updateDeathCapture(entry, action) {
  if (!deathCaseDir) return;
  if (
    entry.episodeLength > 0 &&
    entry.episodeLength % DEATH_CAPTURE_STRIDE === 0
  ) {
    entry.captureAnchors.push({
      episodeLength: entry.episodeLength,
      snapshot: entry.sim.snapshot(),
      previousAction: entry.previousAction,
    });
    while (entry.captureAnchors.length > DEATH_CAPTURE_ANCHORS) {
      entry.captureAnchors.shift();
    }
    const oldestLength = entry.captureAnchors[0].episodeLength;
    entry.captureActions = entry.captureActions.filter(
      (item) => item.episodeLength >= oldestLength,
    );
  }
  entry.captureActions.push({ episodeLength: entry.episodeLength, action });
}

// Tree-search checkpoints. Snapshots stay in this worker; Python only ever
// names a slot. Everything the replay after a rewind depends on is stored,
// including the episode counter that seeds the sim if the branch dies.
const slots = new Map();

function saveSlot(entry, slot) {
  slots.set(slot, {
    snapshot: entry.sim.snapshot(),
    previousAction: entry.previousAction,
    episodeReturn: entry.episodeReturn,
    episodeLength: entry.episodeLength,
    episode: entry.episode,
    startHeight: entry.startHeight,
    curriculumSource: entry.curriculumSource,
    cellVariantId: entry.cellVariantId,
    captureAnchors: entry.captureAnchors.slice(),
    captureActions: entry.captureActions.slice(),
  });
}

function restoreSlot(entry, slot) {
  const saved = slots.get(slot);
  if (!saved) throw new Error(`unknown slot ${slot}`);
  entry.sim.restore(saved.snapshot);
  entry.previousAction = saved.previousAction;
  entry.episodeReturn = saved.episodeReturn;
  entry.episodeLength = saved.episodeLength;
  entry.episode = saved.episode;
  entry.startHeight = saved.startHeight;
  entry.curriculumSource = saved.curriculumSource;
  entry.cellVariantId = saved.cellVariantId;
  entry.captureAnchors = saved.captureAnchors.slice();
  entry.captureActions = saved.captureActions.slice();
}

function isSheltered(sim) {
  const player = sim.player;
  return sim.blocks.blocks.some((block) => {
    if (!block.fixed || block.faultTimer > 0) return false;
    if (block.y + block.h > player.y + 0.001) return false;
    const overlap = Math.min(player.x + player.w, block.x + block.w) -
      Math.max(player.x, block.x);
    return overlap >= 6;
  });
}

for (const entry of envs) resetDeathCapture(entry);

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
  episodeCellIds = null,
  deathCauses = null,
  deathPhases = null,
  deathFocus = null,
  stepPhases = null,
  stepSheltered = null,
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
  offset += count * 4;
  for (let index = 0; index < count; index++) packet.writeInt32LE(episodeCellIds?.[index] ?? -1, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet.writeInt32LE(envs[index].cellVariantId, offset + index * 4);
  offset += count * 4;
  for (let index = 0; index < count; index++) packet[offset + index] = deathCauses?.[index] ?? 0;
  offset += count;
  for (let index = 0; index < count; index++) packet[offset + index] = deathPhases?.[index] ?? 0;
  offset += count;
  for (let index = 0; index < count; index++) packet[offset + index] = deathFocus?.[index] ?? 0;
  offset += count;
  for (let index = 0; index < count; index++) packet[offset + index] = stepPhases?.[index] ?? 0;
  offset += count;
  for (let index = 0; index < count; index++) packet[offset + index] = stepSheltered?.[index] ?? 0;
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

const DEATH_CAUSE = Object.freeze({ fell: 1, squished: 2 });
const PHASE_CODE = Object.freeze({ opening: 1, calm: 2, build: 3, surge: 4, release: 5 });

function writeDeathCase(entry, index, previousAction, action) {
  if (!deathCaseDir || !entry.sim.dead) return;
  const targetLength = Math.max(0, entry.episodeLength - 240);
  const anchor = entry.captureAnchors
    .filter((candidate) => candidate.episodeLength <= targetLength)
    .at(-1) ?? entry.captureAnchors[0];
  const actions = entry.captureActions
    .filter((item) => item.episodeLength >= anchor.episodeLength)
    .map((item) => item.action);
  const payload = {
    version: 2,
    seed: entry.sim.seed,
    workerPid: process.pid,
    environment: index,
    episode: entry.episode,
    frame: entry.sim.frame,
    height: entry.sim.height,
    cause: entry.sim.deathCause,
    phase: entry.sim.director.phase,
    previousAction,
    fatalAction: action,
    history: {
      snapshot: anchor.snapshot,
      previousAction: anchor.previousAction,
      actions: Buffer.from(actions).toString('base64'),
    },
  };
  const filename = `death-${process.pid}-${index}-${entry.episode}-${entry.sim.frame}.json.gz`;
  const target = path.join(deathCaseDir, filename);
  const temporary = `${target}.tmp`;
  fs.writeFileSync(temporary, zlib.gzipSync(JSON.stringify(payload)));
  fs.renameSync(temporary, target);
}

function step(actions, resetIds) {
  const rewards = new Float32Array(count);
  const dones = new Uint8Array(count);
  const successes = new Uint8Array(count);
  const returns = new Float32Array(count);
  const lengths = new Uint32Array(count);
  const heights = new Float32Array(count);
  const episodeStarts = new Float32Array(count);
  const worldScales = new Float32Array(count);
  const episodeCellIds = new Int32Array(count);
  episodeCellIds.fill(-1);
  const deathCauses = new Uint8Array(count);
  const deathPhases = new Uint8Array(count);
  const deathFocus = new Uint8Array(count);
  const stepPhases = new Uint8Array(count);
  const stepSheltered = new Uint8Array(count);
  for (let index = 0; index < count; index++) {
    const entry = envs[index];
    if (actions[index] === 254) continue;
    if (actions[index] === 255) {
      reset(entry, index, resetIds[index]);
      continue;
    }
    if (actions[index] === 253) {
      if (resetIds[index] >= 0) saveSlot(entry, resetIds[index]);
      continue;
    }
    if (actions[index] === 252) {
      if (resetIds[index] >= 0) restoreSlot(entry, resetIds[index]);
      continue;
    }
    const action = Math.min(ACTION_COUNT - 1, actions[index]);
    const previousAction = entry.previousAction;
    stepPhases[index] = PHASE_CODE[entry.sim.director.phase] ?? 0;
    stepSheltered[index] = isSheltered(entry.sim) ? 1 : 0;
    updateDeathCapture(entry, action);
    // The block is one transition: rewards and world time accumulate across it,
    // and it ends early on death or success so neither is stepped past. At
    // gamma = 1 summing undiscounted sub-step rewards is exact; below 1 it
    // slightly under-discounts within the block, which is bounded by the block
    // length and noted rather than corrected.
    let reward = 0;
    let worldScale = 0;
    let success = false;
    for (let repeat = 0; repeat < actionRepeat; repeat++) {
      const beforeStep = entry.sim.height;
      const transition = entry.sim.step(heldActionInput(action, entry.previousAction));
      entry.previousAction = action;
      worldScale += transition.worldScale;
      success = !entry.sim.dead && entry.sim.height >= targetHeight;
      reward += transitionReward(beforeStep, entry.sim, transition.worldScale, success);
      entry.episodeLength++;
      if (entry.sim.dead || success) break;
    }
    worldScales[index] = worldScale;
    entry.episodeReturn += reward;
    rewards[index] = reward;
    if (!entry.sim.dead && !success) continue;
    if (entry.sim.dead) {
      writeDeathCase(entry, index, previousAction, action);
    }
    dones[index] = 1;
    successes[index] = success ? 1 : 0;
    returns[index] = entry.episodeReturn;
    lengths[index] = entry.episodeLength;
    heights[index] = entry.sim.height;
    episodeStarts[index] = entry.startHeight;
    episodeCellIds[index] = entry.cellVariantId;
    deathCauses[index] = DEATH_CAUSE[entry.sim.deathCause] ?? 0;
    deathPhases[index] = PHASE_CODE[entry.sim.director.phase] ?? 0;
    deathFocus[index] = entry.sim.player.focus;
    reset(entry, index, resetIds[index]);
  }
  writePacket(
    rewards,
    dones,
    successes,
    returns,
    lengths,
    heights,
    episodeStarts,
    worldScales,
    episodeCellIds,
    deathCauses,
    deathPhases,
    deathFocus,
    stepPhases,
    stepSheltered,
  );
}

writePacket();
let pending = Buffer.alloc(0);
process.stdin.on('data', (chunk) => {
  pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
  const commandBytes = count * 5;
  while (pending.length >= commandBytes) {
    const actions = pending.subarray(0, count);
    const resetIds = new Int32Array(count);
    for (let index = 0; index < count; index++) {
      resetIds[index] = pending.readInt32LE(count + index * 4);
    }
    pending = pending.subarray(commandBytes);
    step(actions, resetIds);
  }
});

process.on('SIGTERM', () => process.exit(0));
