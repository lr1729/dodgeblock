#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { BLOCK_H, FOCUS_FRAMES, GAME_H } from '../src/constants.js';
import { Sim } from '../src/sim/sim.js';
import {
  FALLING_COUNT,
  FALLING_FEATURES,
  FORECAST_COUNT,
  FORECAST_FEATURES,
  SKYLINE_SIZE,
  STATE_SIZE,
  TERRAIN_COLS,
  TERRAIN_ROWS,
  createObservation,
  encodeObservation,
  heldActionInput,
} from './env-v2.mjs';

function numberArg(name, fallback) {
  const index = process.argv.indexOf(name);
  const value = index === -1 ? fallback : Number(process.argv[index + 1]);
  if (!Number.isFinite(value)) throw new Error(`${name} must be numeric`);
  return value;
}

function stringArg(name, fallback = '') {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : process.argv[index + 1];
}

const demoPath = path.resolve(stringArg('--demo'));
if (!stringArg('--demo')) throw new Error('provide --demo DEMONSTRATION.json.gz');
const outputDir = path.resolve(stringArg('--output-dir', 'rl/counterfactual-v5'));
const shardSeed = Math.floor(numberArg('--shard-seed', 4001));
const searchSeed = numberArg('--search-seed', 0x4354_4654) >>> 0;
const horizon = Math.max(30, Math.floor(numberArg('--horizon', 360)));
const futures = Math.max(1, Math.floor(numberArg('--futures', 2)));
const controlInterval = Math.max(
  1,
  Math.floor(numberArg('--control-interval', 1)),
);
const selectionMode = stringArg('--selection-mode', 'mixed');
if (!['mixed', 'cadence'].includes(selectionMode)) {
  throw new Error('--selection-mode must be mixed or cadence');
}
const branchFrames = Math.max(
  1,
  Math.floor(numberArg(
    '--branch-frames',
    selectionMode === 'cadence' ? controlInterval : 6,
  )),
);
if (selectionMode === 'cadence' && branchFrames !== controlInterval) {
  throw new Error(
    'cadence mode requires --branch-frames to equal --control-interval',
  );
}
const stride = Math.max(1, Math.floor(numberArg('--stride', 60)));
const maxStates = Math.max(1, Math.floor(numberArg('--max-states', 512)));
const temperature = Math.max(1e-3, numberArg('--temperature', 2));
const rules = Object.freeze({ autoGuard: false, checkpoints: false });

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function writeGzip(filename, typed) {
  const bytes = Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);
  fs.writeFileSync(filename, zlib.gzipSync(bytes, { level: 6 }));
  return {
    file: path.basename(filename),
    bytes: typed.byteLength,
    sha256: sha256(bytes),
  };
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

function canonicalAction(sim, action, previousAction) {
  if (action < 9) return action;
  const player = sim.player;
  const aiming = player.focusAimTimer > 0;
  const canPress = player.focus > 0 &&
    player.focusTimer <= 0 &&
    previousAction < 9;
  return aiming || canPress ? action : action % 9;
}

function constantPrefix(action) {
  if (action < 9) return Array(branchFrames).fill(action);
  return [
    ...Array(Math.min(branchFrames, 6)).fill(action),
    ...Array(Math.max(0, branchFrames - 6)).fill(action % 9),
  ];
}

function candidateOptions(sim, baselinePrefix) {
  const options = [{
    action: baselinePrefix[0],
    prefix: baselinePrefix,
    baseline: true,
  }];
  for (let action = 0; action < 9; action++) {
    options.push({ action, prefix: constantPrefix(action), baseline: false });
  }
  for (let horizontal = 0; horizontal < 3; horizontal++) {
    options.push({
      action: horizontal,
      prefix: [
        horizontal,
        ...Array(Math.max(0, branchFrames - 1)).fill(3 + horizontal),
      ],
      baseline: false,
    });
  }
  const player = sim.player;
  if (
    player.focusAimTimer > 0 ||
    (player.focus > 0 && player.focusTimer <= 0)
  ) {
    for (let action = 9; action < 18; action++) {
      options.push({ action, prefix: constantPrefix(action), baseline: false });
    }
  }
  return options;
}

function futureSeed(seed, frame, index) {
  let value = (
    (seed >>> 0) ^
    Math.imul((frame + 1) >>> 0, 0x9e37_79b9) ^
    Math.imul(index >>> 0, 0x85eb_ca6b) ^
    searchSeed
  ) >>> 0;
  value ^= value >>> 16;
  return Math.imul(value, 0x7feb_352d) >>> 0;
}

function rollout(snapshot, previousAction, prefix, suffix, futureIndex, targetHeight) {
  const sim = new Sim(snapshot.seed, { rules });
  sim.restore(snapshot);
  if (futureIndex > 0) {
    sim.seed = futureSeed(snapshot.seed, sim.frame, futureIndex);
    sim.rng.s = sim.seed;
    sim.director.reshuffleMaterialRemainder();
  }
  let prior = previousAction;
  let frames = 0;
  for (const proposed of [...prefix, ...suffix].slice(0, horizon)) {
    const action = canonicalAction(sim, proposed, prior);
    sim.step(heldActionInput(action, prior));
    prior = action;
    frames++;
    if (sim.dead || sim.height >= targetHeight) break;
  }
  return {
    frames,
    height: sim.height,
    focus: sim.player.focus,
    headroom: (GAME_H - (sim.player.y + sim.camY)) / BLOCK_H,
    sheltered: isSheltered(sim),
    reachedTarget: sim.height >= targetHeight,
  };
}

function utility(outcomes, startHeight) {
  const minimumFrames = Math.min(...outcomes.map((outcome) => outcome.frames));
  const meanFrames = outcomes.reduce(
    (sum, outcome) => sum + outcome.frames,
    0,
  ) / outcomes.length;
  const minimumProgress = Math.min(
    ...outcomes.map((outcome) => (outcome.height - startHeight) / BLOCK_H),
  );
  const meanProgress = outcomes.reduce(
    (sum, outcome) => sum + (outcome.height - startHeight) / BLOCK_H,
    0,
  ) / outcomes.length;
  const minimumHeadroom = Math.min(...outcomes.map((outcome) => outcome.headroom));
  const minimumFocus = Math.min(...outcomes.map((outcome) => outcome.focus));
  const sheltered = outcomes.filter((outcome) => outcome.sheltered).length /
    outcomes.length;
  return (
    minimumFrames +
    meanFrames * 0.05 +
    minimumProgress * 0.5 +
    meanProgress * 0.1 +
    minimumHeadroom * 0.1 +
    minimumFocus * 0.25 +
    sheltered * 0.5 +
    (outcomes.every((outcome) => outcome.reachedTarget) ? 1_000_000 : 0)
  );
}

function selectedFrames(actions) {
  if (selectionMode === 'cadence') {
    const frames = [];
    for (let frame = 0; frame < actions.length; frame += controlInterval) {
      frames.push(frame);
    }
    if (frames.length <= maxStates) return new Set(frames);
    const selected = new Set();
    for (let index = 0; index < maxStates; index++) {
      selected.add(frames[Math.floor(index * frames.length / maxStates)]);
    }
    return selected;
  }
  const candidates = new Set([0]);
  let previousAction = 0;
  for (let frame = 0; frame < actions.length; frame++) {
    const action = actions[frame];
    if (
      frame % stride === 0 ||
      action !== previousAction ||
      (action >= 9 && previousAction < 9)
    ) {
      candidates.add(frame);
    }
    previousAction = action;
  }
  const frames = [...candidates].sort((left, right) => left - right);
  if (frames.length <= maxStates) return new Set(frames);
  const selected = new Set();
  for (let index = 0; index < maxStates; index++) {
    selected.add(frames[Math.floor(index * frames.length / maxStates)]);
  }
  return selected;
}

function normalizedTargets(scored) {
  const bestByAction = new Map();
  for (const candidate of scored) {
    const incumbent = bestByAction.get(candidate.action);
    if (!incumbent || candidate.score > incumbent.score) {
      bestByAction.set(candidate.action, candidate);
    }
  }
  const candidates = [...bestByAction.values()];
  const maximum = Math.max(...candidates.map((candidate) => candidate.score));
  const weights = candidates.map((candidate) =>
    Math.exp((candidate.score - maximum) / temperature));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const targets = new Float32Array(18);
  for (let index = 0; index < candidates.length; index++) {
    targets[candidates[index].action] = weights[index] / total;
  }
  return targets;
}

const compressedDemo = fs.readFileSync(demoPath);
const payload = JSON.parse(zlib.gunzipSync(compressedDemo));
if (![1, 2].includes(payload.version)) {
  throw new Error(`unsupported demonstration version: ${payload.version}`);
}
const actions = Uint8Array.from(Buffer.from(payload.actions, 'base64'));
if (actions.length !== payload.frames) throw new Error('demo action length mismatch');
if (horizon <= branchFrames) throw new Error('--horizon must exceed --branch-frames');
const frameSelection = selectedFrames(actions);
const sim = new Sim(payload.seed, { rules });
if (payload.version === 2) sim.restore(payload.initialSnapshot);
let previousAction = payload.initialPreviousAction ?? 0;
const observation = createObservation();
const records = [];
const diagnostics = [];
const started = Date.now();

for (let frame = 0; frame < actions.length; frame++) {
  if (frameSelection.has(frame)) {
    const available = Math.min(branchFrames, actions.length - frame);
    const baselinePrefix = selectionMode === 'cadence'
      ? constantPrefix(actions[frame]).slice(0, available)
      : [...actions.slice(frame, frame + available)];
    const suffix = [...actions.slice(
      frame + available,
      Math.min(actions.length, frame + horizon),
    )];
    const snapshot = sim.snapshot();
    const scored = candidateOptions(sim, baselinePrefix).map((candidate) => {
      const outcomes = Array.from({ length: futures }, (_, futureIndex) =>
        rollout(
          snapshot,
          previousAction,
          candidate.prefix,
          suffix,
          futureIndex,
          payload.targetHeight,
        ));
      return {
        ...candidate,
        outcomes,
        score: utility(outcomes, sim.height),
      };
    });
    const baseline = scored.find((candidate) => candidate.baseline);
    if (
      selectionMode === 'mixed' &&
      baseline.outcomes[0].frames < Math.min(horizon, actions.length - frame) &&
      !baseline.outcomes[0].reachedTarget
    ) {
      throw new Error(`successful baseline failed at source frame ${frame}`);
    }
    scored.sort((left, right) =>
      right.score - left.score || left.action - right.action);
    const targets = normalizedTargets(scored);
    encodeObservation(sim, { previousAction }, observation);
    records.push({
      terrain: Uint16Array.from(observation.terrain),
      skyline: Uint8Array.from(observation.skyline),
      falling: Float32Array.from(observation.falling),
      forecasts: Float32Array.from(observation.forecasts),
      state: Float32Array.from(observation.state),
      action: targets.indexOf(Math.max(...targets)),
      targets,
    });
    diagnostics.push({
      frame,
      height: sim.height,
      baselineAction: baseline.action,
      selectedAction: scored[0].action,
      baselineScore: baseline.score,
      selectedScore: scored[0].score,
      entropy: -targets.reduce(
        (sum, probability) =>
          probability > 0 ? sum + probability * Math.log(probability) : sum,
        0,
      ),
    });
  }

  const action = actions[frame];
  sim.step(heldActionInput(action, previousAction));
  previousAction = action;
  if (sim.dead) throw new Error(`demonstration died at source frame ${frame}`);
}
if (sim.height < payload.targetHeight || sim.hash() !== payload.finalHash) {
  throw new Error('demonstration failed final replay verification');
}

const frames = records.length;
const arrays = {
  terrain: new Uint16Array(frames * TERRAIN_ROWS * TERRAIN_COLS),
  skyline: new Uint8Array(frames * SKYLINE_SIZE),
  falling: new Float32Array(frames * FALLING_COUNT * FALLING_FEATURES),
  forecasts: new Float32Array(frames * FORECAST_COUNT * FORECAST_FEATURES),
  state: new Float32Array(frames * STATE_SIZE),
  actions: new Uint8Array(frames),
  targets: new Float32Array(frames * 18),
};
for (let frame = 0; frame < frames; frame++) {
  const record = records[frame];
  arrays.terrain.set(record.terrain, frame * TERRAIN_ROWS * TERRAIN_COLS);
  arrays.skyline.set(record.skyline, frame * SKYLINE_SIZE);
  arrays.falling.set(
    record.falling,
    frame * FALLING_COUNT * FALLING_FEATURES,
  );
  arrays.forecasts.set(
    record.forecasts,
    frame * FORECAST_COUNT * FORECAST_FEATURES,
  );
  arrays.state.set(record.state, frame * STATE_SIZE);
  arrays.actions[frame] = record.action;
  arrays.targets.set(record.targets, frame * 18);
}

const shardDir = path.join(outputDir, `seed-${shardSeed}`);
fs.mkdirSync(shardDir, { recursive: true });
const files = {
  terrain: writeGzip(path.join(shardDir, 'terrain.u16.gz'), arrays.terrain),
  skyline: writeGzip(path.join(shardDir, 'skyline.u8.gz'), arrays.skyline),
  falling: writeGzip(path.join(shardDir, 'falling.f32.gz'), arrays.falling),
  forecasts: writeGzip(path.join(shardDir, 'forecasts.f32.gz'), arrays.forecasts),
  state: writeGzip(path.join(shardDir, 'state.f32.gz'), arrays.state),
  actions: writeGzip(path.join(shardDir, 'actions.u8.gz'), arrays.actions),
  targets: writeGzip(path.join(shardDir, 'targets.f32.gz'), arrays.targets),
};
const mean = (key) => diagnostics.reduce(
  (sum, item) => sum + item[key],
  0,
) / diagnostics.length;
const manifest = {
  version: 1,
  seed: shardSeed,
  frames,
  targetHeight: payload.targetHeight,
  source: 'successful-suffix-counterfactual-v5',
  sourceSeed: payload.seed,
  horizon,
  futures,
  branchFrames,
  controlInterval,
  selectionMode,
  stride,
  temperature,
  meanAdvantage: mean('selectedScore') - mean('baselineScore'),
  meanTargetEntropy: mean('entropy'),
  selectedBaselineFraction: diagnostics.filter(
    (item) => item.selectedAction === item.baselineAction,
  ).length / diagnostics.length,
  demo: {
    file: demoPath,
    sha256: sha256(compressedDemo),
  },
  observation: {
    terrain: [TERRAIN_ROWS, TERRAIN_COLS],
    skyline: [SKYLINE_SIZE],
    falling: [FALLING_COUNT, FALLING_FEATURES],
    forecasts: [FORECAST_COUNT, FORECAST_FEATURES],
    state: [STATE_SIZE],
  },
  files,
};
fs.writeFileSync(
  path.join(shardDir, 'manifest.json'),
  `${JSON.stringify(manifest, null, 2)}\n`,
);
fs.writeFileSync(
  path.join(shardDir, 'diagnostics.json'),
  `${JSON.stringify(diagnostics)}\n`,
);
process.stdout.write(`${JSON.stringify({
  event: 'complete',
  sourceSeed: payload.seed,
  shardSeed,
  frames,
  meanAdvantage: manifest.meanAdvantage,
  meanTargetEntropy: manifest.meanTargetEntropy,
  selectedBaselineFraction: manifest.selectedBaselineFraction,
  controlInterval,
  selectionMode,
  elapsedSeconds: Math.round((Date.now() - started) / 100) / 10,
  output: shardDir,
})}\n`);
