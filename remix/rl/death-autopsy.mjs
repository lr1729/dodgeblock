#!/usr/bin/env node
// When does a death become unavoidable?
//
// Four reward-side interventions have now failed to move the per-layer survival
// plateau. Before designing a fifth, measure the thing none of them assumed:
// at the moment the agent dies, how long had it already been doomed?
//
// Method: replay each death trace exactly, snapshot the sim K frames before the
// death, and search for ANY action sequence that survives H more frames. The
// sim is deterministic and `restore` is exact, so every rollout from a snapshot
// faces the identical falling blocks -- this asks "was there an escape from
// THIS situation", not "is the situation usually escapable".
//
// Read the result in one direction only. A found escape PROVES the state was
// viable. No escape found does not prove doom; it bounds it. So viability(K) is
// a lower bound on how recoverable the agent's position was.
//
// usage: node rl/death-autopsy.mjs deaths.jsonl [--samples 64] [--horizon 120]
import fs from 'node:fs';

import { Sim } from '../src/sim/sim.js';
import { heldActionInput } from './env-v2.mjs';

const REWINDS = [2, 5, 10, 20, 40, 80, 160];
const RULES = { autoGuard: false, checkpoints: false };

function parseFlag(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index < 0 ? fallback : Number(process.argv[index + 1]);
}

const samples = parseFlag('samples', 64);
const horizon = parseFlag('horizon', 120);
const limit = parseFlag('limit', 200);
const traceFile = process.argv[2];

function traceSeed(trace) {
  return trace.bridge_seed === undefined
    ? trace.seed
    : (trace.bridge_seed + trace.env_index * 0x9e3779b9 + trace.episode * 0x85ebca6b) >>> 0;
}

// A sticky random controller. Pure iid actions are not a controller at all --
// they jitter in place and never travel, so they would understate viability.
// Holding each action for a geometric run length makes the search explore
// actual movement, which is what an escape requires.
function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function survives(sim, snapshot, previousAction, random) {
  sim.restore(snapshot);
  let previous = previousAction;
  let action = Math.floor(random() * 18);
  let hold = 0;
  for (let frame = 0; frame < horizon; frame++) {
    if (hold <= 0) {
      action = Math.floor(random() * 18);
      hold = 1 + Math.floor(random() * 14);
    }
    hold--;
    sim.step(heldActionInput(action, previous));
    previous = action;
    if (sim.dead) return false;
  }
  return true;
}

const traces = fs.readFileSync(traceFile, 'utf8').trim().split('\n')
  .map((line) => JSON.parse(line))
  .filter((trace) => trace.outcome === 'death' && trace.frames > 200)
  .slice(0, limit);

const viable = new Map(REWINDS.map((k) => [k, { yes: 0, total: 0 }]));
const causes = {};
const escapeDepth = [];

for (const trace of traces) {
  const seed = traceSeed(trace);
  const actions = Buffer.from(trace.actions, 'base64');
  const sim = new Sim(seed, { rules: RULES });
  const wanted = new Map();
  for (const k of REWINDS) {
    if (trace.frames - k > 0) wanted.set(trace.frames - k, k);
  }

  // One forward replay, snapshotting at each rewind offset.
  const points = new Map();
  let previous = 0;
  for (let frame = 0; frame < trace.frames; frame++) {
    if (wanted.has(frame)) {
      points.set(wanted.get(frame), { snapshot: sim.snapshot(), previous });
    }
    sim.step(heldActionInput(actions[frame], previous));
    previous = actions[frame];
    if (sim.dead) break;
  }
  causes[sim.deathCause ?? 'unknown'] = (causes[sim.deathCause ?? 'unknown'] ?? 0) + 1;

  const scratch = new Sim(seed, { rules: RULES });
  let deepest = 0;
  for (const k of REWINDS) {
    const point = points.get(k);
    if (!point) continue;
    const record = viable.get(k);
    record.total++;
    const random = makeRandom(seed ^ (k * 0x9e3779b9));
    let escaped = false;
    for (let sample = 0; sample < samples && !escaped; sample++) {
      escaped = survives(scratch, point.snapshot, point.previous, random);
    }
    if (escaped) { record.yes++; deepest = Math.max(deepest, k); }
  }
  escapeDepth.push(deepest);
}

escapeDepth.sort((a, b) => a - b);
const median = escapeDepth[Math.floor(escapeDepth.length / 2)] ?? 0;
console.log(JSON.stringify({
  deaths: traces.length,
  samples,
  horizon,
  viability: Object.fromEntries(REWINDS.map((k) => {
    const record = viable.get(k);
    return [k, record.total ? +(record.yes / record.total).toFixed(3) : null];
  })),
  median_escape_depth: median,
  causes,
}, null, 2));
