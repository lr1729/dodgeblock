#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

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

function stringArg(name, fallback = '') {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : process.argv[index + 1];
}

function numberArg(name, fallback) {
  const value = Number(stringArg(name, fallback));
  if (!Number.isFinite(value)) throw new Error(`${name} must be numeric`);
  return value;
}

function repeatedArgs(name) {
  const values = [];
  for (let index = 0; index < process.argv.length - 1; index++) {
    if (process.argv[index] === name) values.push(path.resolve(process.argv[index + 1]));
  }
  return values;
}

function inputFiles(inputs) {
  return inputs.flatMap((input) => {
    const stat = fs.statSync(input);
    return stat.isDirectory()
      ? fs.readdirSync(input)
        .filter((filename) => filename.endsWith('.json'))
        .sort()
        .map((filename) => path.join(input, filename))
      : [input];
  });
}

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

function rewindView(deathCase, rewind) {
  if (deathCase.version !== 2) {
    throw new Error(`correction export requires death case version 2`);
  }
  const actions = [...Buffer.from(deathCase.history.actions, 'base64')];
  if (rewind > actions.length) return null;
  const prefixLength = actions.length - rewind;
  const rules = Object.freeze({ autoGuard: false, checkpoints: false });
  const sim = new Sim(deathCase.history.snapshot.seed, { rules });
  sim.restore(deathCase.history.snapshot);
  let previousAction = deathCase.history.previousAction;
  for (const action of actions.slice(0, prefixLength)) {
    sim.step(heldActionInput(action, previousAction));
    previousAction = action;
    if (sim.dead) throw new Error('death history terminates before rewind');
  }
  return { sim, previousAction };
}

function copyObservation(target, frame, observation) {
  target.terrain.set(observation.terrain, frame * observation.terrain.length);
  target.skyline.set(observation.skyline, frame * observation.skyline.length);
  target.falling.set(observation.falling, frame * observation.falling.length);
  target.forecasts.set(observation.forecasts, frame * observation.forecasts.length);
  target.state.set(observation.state, frame * observation.state.length);
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

const inputs = repeatedArgs('--oracle');
if (!inputs.length) throw new Error('provide at least one --oracle file or directory');
const outputDir = path.resolve(stringArg('--output-dir', 'rl/oracle-corrections-v5'));
const shardSeed = Math.floor(numberArg('--shard-seed', 1001));
const prefixFrames = Math.max(1, Math.floor(numberArg('--prefix-frames', 60)));
const records = [];
const sourceHashes = [];

for (const filename of inputFiles(inputs)) {
  const bytes = fs.readFileSync(filename);
  sourceHashes.push(sha256(bytes));
  const oracle = JSON.parse(bytes);
  for (const entry of oracle.cases ?? []) {
    if (!entry.robust?.actions) continue;
    const deathCase = JSON.parse(zlib.gunzipSync(fs.readFileSync(entry.filename)));
    const view = rewindView(deathCase, entry.rewind);
    if (!view) continue;
    records.push({
      filename: entry.filename,
      rewind: entry.rewind,
      sim: view.sim,
      previousAction: view.previousAction,
      actions: [...Buffer.from(entry.robust.actions, 'base64')].slice(0, prefixFrames),
    });
  }
}
if (!records.length) throw new Error('oracle inputs contain no robust corrections');

const frames = records.reduce((total, record) => total + record.actions.length, 0);
const arrays = {
  terrain: new Uint16Array(frames * TERRAIN_ROWS * TERRAIN_COLS),
  skyline: new Uint8Array(frames * SKYLINE_SIZE),
  falling: new Float32Array(frames * FALLING_COUNT * FALLING_FEATURES),
  forecasts: new Float32Array(frames * FORECAST_COUNT * FORECAST_FEATURES),
  state: new Float32Array(frames * STATE_SIZE),
  actions: new Uint8Array(frames),
};
const observation = createObservation();
let frame = 0;
for (const record of records) {
  let previousAction = record.previousAction;
  for (const proposedAction of record.actions) {
    const action = canonicalAction(record.sim, proposedAction, previousAction);
    encodeObservation(record.sim, { previousAction }, observation);
    copyObservation(arrays, frame, observation);
    arrays.actions[frame] = action;
    record.sim.step(heldActionInput(action, previousAction));
    previousAction = action;
    if (record.sim.dead) {
      throw new Error(`robust correction died in original future: ${record.filename}`);
    }
    frame++;
  }
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
};
const sourceDigest = sha256(Buffer.from(sourceHashes.sort().join('\n')));
const manifest = {
  version: 1,
  seed: shardSeed,
  frames,
  targetHeight: 10_000,
  source: 'on-policy-robust-oracle-v5',
  correctionCases: records.length,
  prefixFrames,
  demo: {
    file: 'oracle-results',
    sha256: sourceDigest,
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
process.stdout.write(`${JSON.stringify({
  event: 'exported',
  shardSeed,
  cases: records.length,
  frames,
  output: shardDir,
})}\n`);
