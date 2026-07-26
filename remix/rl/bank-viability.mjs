#!/usr/bin/env node
// Is the saturated regime survivable at the rate 10k requires?
//
// The policy holds per-layer survival 0.694 from banked saturated states and 20M
// frames of training there did not move it. Before concluding the policy class is
// the problem, establish the ceiling: how survivable are these states at all?
//
// From each banked cell at frame >= 14400 (240 s, past difficulty saturation),
// run K sticky-random rollouts for a fixed window and report:
//
//   - per-rollout survival: what a memoryless flailing controller achieves
//   - best-of-K survival:   whether an escape EXISTS from the state, which is a
//                           lower bound on viability (a found escape proves it)
//
// The window is 240 frames = 4 s, roughly the time competent play takes to gain
// one layer at the measured 10.5 height/s, so per-rollout survival is directly
// comparable to the per-layer survival figures in the ledger.
//
// Read best-of-K as an upper bound on nothing: it is a 64-retry statistic, and a
// live policy gets one attempt. It answers only "was this state survivable",
// which is the question that decides whether the goal is reachable in principle.
//
// usage: node rl/bank-viability.mjs <search-checkpoint.json.gz> [--cells 200]
import fs from 'node:fs';
import zlib from 'node:zlib';

import { Sim } from '../src/sim/sim.js';
import { heldActionInput } from './env-v2.mjs';

const RULES = { autoGuard: false, checkpoints: false };
const SATURATION_FRAME = 14400;

function flag(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index < 0 ? fallback : Number(process.argv[index + 1]);
}

const bankFile = process.argv[2];
const cellCount = flag('cells', 200);
const samples = flag('samples', 64);
const window = flag('window', 240);

const bank = JSON.parse(zlib.gunzipSync(fs.readFileSync(bankFile)));
const saturated = bank.entries.filter((entry) => entry.frame >= SATURATION_FRAME);

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

// Sticky random: mean hold ~7.5 frames. Pure iid actions never travel, so they
// would understate what a memoryless controller can do.
function rollout(sim, snapshot, previousAction, random) {
  sim.restore(snapshot);
  let previous = previousAction;
  let action = Math.floor(random() * 18);
  let hold = 0;
  const startHeight = sim.height;
  for (let frame = 0; frame < window; frame++) {
    if (hold <= 0) {
      action = Math.floor(random() * 18);
      hold = 1 + Math.floor(random() * 14);
    }
    hold--;
    sim.step(heldActionInput(action, previous));
    previous = action;
    if (sim.dead) return { survived: false, gained: sim.height - startHeight };
  }
  return { survived: true, gained: sim.height - startHeight };
}

const stride = Math.max(1, Math.floor(saturated.length / cellCount));
const chosen = saturated.filter((_, index) => index % stride === 0).slice(0, cellCount);

let survivals = 0;
let attempts = 0;
let viable = 0;
let gainedTotal = 0;
const scratch = new Sim(1, { rules: RULES });

for (const [index, cell] of chosen.entries()) {
  const random = makeRandom((bank.seed ?? 1) ^ (index * 0x9e3779b9));
  let anySurvived = false;
  for (let sample = 0; sample < samples; sample++) {
    const result = rollout(scratch, cell.snapshot, cell.previousAction ?? 0, random);
    attempts++;
    if (result.survived) { survivals++; anySurvived = true; }
    gainedTotal += result.gained;
  }
  if (anySurvived) viable++;
}

const perRollout = survivals / attempts;
console.log(JSON.stringify({
  bank: bankFile,
  saturated_cells_available: saturated.length,
  cells_probed: chosen.length,
  samples_per_cell: samples,
  window_frames: window,
  sticky_random_survival: +perRollout.toFixed(4),
  fraction_of_cells_with_an_escape: +(viable / chosen.length).toFixed(4),
  mean_height_gained_per_rollout: +(gainedTotal / attempts).toFixed(1),
  reference: {
    policy_at_saturated_cells: 0.6941,
    needed_to_reach_10k_at_all: 0.9567,
    needed_for_consistent_10k: 0.9972,
  },
  reading: 'per-rollout survival is what a memoryless flailing controller gets '
    + 'over one layer-time; fraction_with_an_escape proves survivability but is a '
    + 'retry statistic and is not achievable by a policy that gets one attempt',
}, null, 2));
