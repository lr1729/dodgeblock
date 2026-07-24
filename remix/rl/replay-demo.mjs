#!/usr/bin/env node
import fs from 'node:fs';
import zlib from 'node:zlib';

import { Sim } from '../src/sim/sim.js';
import { heldActionInput } from './env-v2.mjs';

const filename = process.argv[2];
if (!filename) throw new Error('usage: node rl/replay-demo.mjs DEMONSTRATION.json.gz');

const payload = JSON.parse(zlib.gunzipSync(fs.readFileSync(filename)));
if (![1, 2].includes(payload.version)) {
  throw new Error(`unsupported demonstration version: ${payload.version}`);
}
const actions = Buffer.from(payload.actions, 'base64');
if (actions.length !== payload.frames) {
  throw new Error(`action length mismatch: expected ${payload.frames}, got ${actions.length}`);
}

const sim = new Sim(payload.seed, {
  rules: { autoGuard: false, checkpoints: false },
});
if (payload.version === 2) sim.restore(payload.initialSnapshot);
let previousAction = payload.initialPreviousAction ?? 0;
for (const action of actions) {
  sim.step(heldActionInput(action, previousAction));
  previousAction = action;
  if (sim.dead) {
    throw new Error(`demonstration died at frame ${sim.frame}, height ${sim.height}: ${sim.deathCause}`);
  }
}

if (sim.height < payload.targetHeight) {
  throw new Error(`demonstration ended at ${sim.height}, below target ${payload.targetHeight}`);
}
if (payload.finalHash && sim.hash() !== payload.finalHash) {
  throw new Error('demonstration final state hash does not match');
}

process.stdout.write(`${JSON.stringify({
  seed: payload.seed,
  frames: actions.length,
  height: sim.height,
  targetHeight: payload.targetHeight,
  finalHash: sim.hash(),
})}\n`);
