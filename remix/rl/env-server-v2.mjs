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
const rewardMode = stringArg('--reward-mode', 'height');
if (!['height', 'target'].includes(rewardMode)) {
  throw new Error(`unsupported reward mode: ${rewardMode}`);
}
const rules = Object.freeze({ autoGuard: false, checkpoints: false });
const cellBankPaths = repeatedArgs('--cell-bank');
const deathCaseDir = stringArg('--death-case-dir', '');
if (deathCaseDir) fs.mkdirSync(deathCaseDir, { recursive: true });
const observationBytes = observationByteSize();
const statsBytes = count * (4 + 1 + 1 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 1 + 1 + 1);
const packetBytes = count * observationBytes + statsBytes;

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
  episodeCellIds = null,
  deathCauses = null,
  deathPhases = null,
  deathFocus = null,
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

function writeDeathCase(entry, index, snapshot, previousAction, action) {
  if (!deathCaseDir || !entry.sim.dead) return;
  const payload = {
    version: 1,
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
    snapshot,
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
  for (let index = 0; index < count; index++) {
    const entry = envs[index];
    if (actions[index] === 254) continue;
    if (actions[index] === 255) {
      reset(entry, index, resetIds[index]);
      continue;
    }
    const beforeHeight = entry.sim.height;
    const action = Math.min(ACTION_COUNT - 1, actions[index]);
    const previousAction = entry.previousAction;
    const beforeSnapshot = deathCaseDir ? entry.sim.snapshot() : null;
    const transition = entry.sim.step(heldActionInput(action, entry.previousAction));
    const success = !entry.sim.dead && entry.sim.height >= targetHeight;
    worldScales[index] = transition.worldScale;
    entry.previousAction = action;
    const reward = transitionReward(beforeHeight, entry.sim, transition.worldScale, success);
    entry.episodeReturn += reward;
    entry.episodeLength++;
    rewards[index] = reward;
    if (!entry.sim.dead && !success) continue;
    if (entry.sim.dead) {
      writeDeathCase(entry, index, beforeSnapshot, previousAction, action);
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
