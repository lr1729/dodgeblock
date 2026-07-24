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

function parseArguments() {
  const result = {
    demos: [],
    outputDir: path.resolve('rl/demo-dataset-v5'),
    snapshotInterval: 120,
  };
  for (let index = 2; index < process.argv.length; index++) {
    const value = process.argv[index];
    if (value === '--demo') result.demos.push(path.resolve(process.argv[++index]));
    else if (value === '--output-dir') result.outputDir = path.resolve(process.argv[++index]);
    else if (value === '--snapshot-interval') {
      result.snapshotInterval = Math.max(1, Number(process.argv[++index]) | 0);
    } else {
      throw new Error(`unknown argument: ${value}`);
    }
  }
  if (!result.demos.length) throw new Error('at least one --demo is required');
  return result;
}

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function writeGzip(filename, typed) {
  const bytes = Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);
  const temporary = `${filename}.tmp`;
  fs.writeFileSync(temporary, zlib.gzipSync(bytes, { level: 6 }));
  fs.renameSync(temporary, filename);
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

function terrainRelief(sim) {
  const layers = sim.blocks.surfaceLayers();
  return layers.length ? Math.max(...layers) - Math.min(...layers) : 0;
}

function phaseCode(phase) {
  return ['opening', 'calm', 'build', 'surge', 'release'].indexOf(phase) + 1;
}

function copyObservation(target, frame, observation) {
  target.terrain.set(observation.terrain, frame * observation.terrain.length);
  target.skyline.set(observation.skyline, frame * observation.skyline.length);
  target.falling.set(observation.falling, frame * observation.falling.length);
  target.forecasts.set(observation.forecasts, frame * observation.forecasts.length);
  target.state.set(observation.state, frame * observation.state.length);
}

function exportDemo(filename, outputDir, snapshotInterval) {
  const compressed = fs.readFileSync(filename);
  const payload = JSON.parse(zlib.gunzipSync(compressed));
  if (payload.version !== 1) throw new Error(`unsupported demo version in ${filename}`);
  const actions = Buffer.from(payload.actions, 'base64');
  if (actions.length !== payload.frames) {
    throw new Error(`action length mismatch in ${filename}`);
  }

  const frames = actions.length;
  const arrays = {
    terrain: new Uint16Array(frames * TERRAIN_ROWS * TERRAIN_COLS),
    skyline: new Uint8Array(frames * SKYLINE_SIZE),
    falling: new Float32Array(frames * FALLING_COUNT * FALLING_FEATURES),
    forecasts: new Float32Array(frames * FORECAST_COUNT * FORECAST_FEATURES),
    state: new Float32Array(frames * STATE_SIZE),
    actions: new Uint8Array(actions),
    heights: new Float32Array(frames),
    phases: new Uint8Array(frames),
    sheltered: new Uint8Array(frames),
    relief: new Uint16Array(frames),
  };
  const sim = new Sim(payload.seed, {
    rules: Object.freeze({ autoGuard: false, checkpoints: false }),
  });
  const observation = createObservation();
  let previousAction = 0;
  const snapshots = [];
  const bandStats = new Map();

  for (let frame = 0; frame < frames; frame++) {
    encodeObservation(sim, { previousAction }, observation);
    copyObservation(arrays, frame, observation);
    arrays.heights[frame] = sim.height;
    arrays.phases[frame] = phaseCode(sim.director.phase);
    arrays.sheltered[frame] = isSheltered(sim) ? 1 : 0;
    arrays.relief[frame] = Math.max(0, Math.min(65535, terrainRelief(sim)));
    const band = Math.floor(sim.height / 400) * 400;
    const stats = bandStats.get(band) ?? { frames: 0, sheltered: 0, relief: 0 };
    stats.frames++;
    stats.sheltered += arrays.sheltered[frame];
    stats.relief += arrays.relief[frame];
    bandStats.set(band, stats);

    if (frame % snapshotInterval === 0) {
      snapshots.push({
        key: `trajectory|${payload.seed}|${frame}`,
        snapshot: sim.snapshot(),
        previousAction,
        height: sim.height,
        frame,
      });
    }
    const action = actions[frame];
    sim.step(heldActionInput(action, previousAction));
    previousAction = action;
    if (sim.dead) throw new Error(`demo ${filename} dies at frame ${frame}`);
  }
  if (sim.height < payload.targetHeight || sim.hash() !== payload.finalHash) {
    throw new Error(`demo ${filename} does not replay to its declared terminal state`);
  }

  const seedDir = path.join(outputDir, `seed-${payload.seed}`);
  fs.mkdirSync(seedDir, { recursive: true });
  const files = {
    terrain: writeGzip(path.join(seedDir, 'terrain.u16.gz'), arrays.terrain),
    skyline: writeGzip(path.join(seedDir, 'skyline.u8.gz'), arrays.skyline),
    falling: writeGzip(path.join(seedDir, 'falling.f32.gz'), arrays.falling),
    forecasts: writeGzip(path.join(seedDir, 'forecasts.f32.gz'), arrays.forecasts),
    state: writeGzip(path.join(seedDir, 'state.f32.gz'), arrays.state),
    actions: writeGzip(path.join(seedDir, 'actions.u8.gz'), arrays.actions),
    heights: writeGzip(path.join(seedDir, 'heights.f32.gz'), arrays.heights),
    phases: writeGzip(path.join(seedDir, 'phases.u8.gz'), arrays.phases),
    sheltered: writeGzip(path.join(seedDir, 'sheltered.u8.gz'), arrays.sheltered),
    relief: writeGzip(path.join(seedDir, 'relief.u16.gz'), arrays.relief),
  };
  const bank = {
    version: 1,
    targetHeight: payload.targetHeight,
    seed: payload.seed,
    source: 'successful-trajectory-v5',
    entries: snapshots,
  };
  const bankBytes = Buffer.from(JSON.stringify(bank));
  const bankFile = path.join(seedDir, 'trajectory-bank.json.gz');
  fs.writeFileSync(bankFile, zlib.gzipSync(bankBytes));

  const manifest = {
    version: 1,
    seed: payload.seed,
    frames,
    targetHeight: payload.targetHeight,
    finalHeight: sim.height,
    finalHash: payload.finalHash,
    demo: {
      file: path.resolve(filename),
      sha256: sha256(compressed),
    },
    observation: {
      terrain: [TERRAIN_ROWS, TERRAIN_COLS],
      skyline: [SKYLINE_SIZE],
      falling: [FALLING_COUNT, FALLING_FEATURES],
      forecasts: [FORECAST_COUNT, FORECAST_FEATURES],
      state: [STATE_SIZE],
    },
    files,
    snapshotInterval,
    trajectoryBank: {
      file: path.basename(bankFile),
      entries: snapshots.length,
      sha256: sha256(bankBytes),
    },
    actionCounts: [...arrays.actions.reduce((counts, action) => {
      counts[action] = (counts[action] ?? 0) + 1;
      return counts;
    }, Array(18).fill(0))],
    bands: [...bandStats.entries()].map(([height, stats]) => ({
      height,
      frames: stats.frames,
      shelteredFraction: stats.sheltered / stats.frames,
      meanReliefLayers: stats.relief / stats.frames,
    })),
  };
  fs.writeFileSync(
    path.join(seedDir, 'manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  process.stdout.write(`${JSON.stringify({
    event: 'exported',
    seed: payload.seed,
    frames,
    snapshots: snapshots.length,
    output: seedDir,
  })}\n`);
  return manifest;
}

const args = parseArguments();
fs.mkdirSync(args.outputDir, { recursive: true });
args.demos
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
  .map((filename) => exportDemo(filename, args.outputDir, args.snapshotInterval));
const manifests = fs.readdirSync(args.outputDir)
  .filter((name) => /^seed-\d+$/.test(name))
  .map((name) => JSON.parse(
    fs.readFileSync(path.join(args.outputDir, name, 'manifest.json')),
  ))
  .sort((a, b) => a.seed - b.seed);
fs.writeFileSync(
  path.join(args.outputDir, 'dataset.json'),
  `${JSON.stringify({
    version: 1,
    seeds: manifests.map((manifest) => manifest.seed),
    frames: manifests.reduce((sum, manifest) => sum + manifest.frames, 0),
  }, null, 2)}\n`,
);
