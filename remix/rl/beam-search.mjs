#!/usr/bin/env node
// Option-level beam search over the exact simulator.
//
// This is deliberately NOT a TAS. Candidates are scored across reseeded
// futures (`--futures`), so a plan that only works against one known spawn
// sequence is rejected — the search may use no information the live policy
// lacks. That keeps its decisions learnable, which is the point: the search
// exists to be distilled, not to be admired.
//
// Decisions are made at option granularity (a held action for tens of frames)
// because held-out direction loss at 2-frame commitment measured ~ln(3): this
// game's decisions are sparse in time, so per-frame search is mostly noise.
import fs from 'node:fs';
import zlib from 'node:zlib';

import { Sim } from '../src/sim/sim.js';
import { heldActionInput } from './env-v2.mjs';

function parseArguments(argv) {
  const options = {
    seed: 7, target: 10000, beam: 8, horizon: 45, futures: 2,
    maxFrames: 40000, output: '', progressEvery: 20, focusPress: 4,
    deathPenalty: 400,
  };
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index].replace(/^--/, '');
    const value = argv[index + 1];
    if (!(key in options)) throw new Error(`unknown option --${key}`);
    options[key] = typeof options[key] === 'number' ? Number(value) : value;
  }
  return options;
}

const config = parseArguments(process.argv.slice(2));

// A move is a short action program: hold `action` for `frames`, optionally
// preceded by a focus press so that press->release commits a dash.
function moves(sim) {
  const list = [];
  for (let action = 0; action < 9; action++) {
    list.push({ label: `hold-${action}`, frames: [[action, config.horizon]] });
  }
  const player = sim.player;
  const canFocus = player.focusAimTimer > 0 ||
    (player.focus > 0 && player.focusTimer <= 0);
  if (canFocus) {
    for (let action = 0; action < 9; action++) {
      list.push({
        label: `focus-${action}`,
        frames: [
          [action + 9, config.focusPress],
          [action, Math.max(1, config.horizon - config.focusPress)],
        ],
      });
    }
  }
  return list;
}

function expand(move) {
  const actions = [];
  for (const [action, count] of move.frames) {
    for (let index = 0; index < count; index++) actions.push(action);
  }
  return actions;
}

function futureSeed(seed, index) {
  return (Math.imul(seed ^ (index + 1), 0x9e3779b1) >>> 0);
}

// Play `actions` from `snapshot`; futureIndex 0 keeps the real future, others
// reseed the director so a candidate cannot exploit one spawn sequence.
function play(snapshot, previousAction, actions, seed, futureIndex) {
  const sim = new Sim(seed, { rules: { autoGuard: false, checkpoints: false } });
  sim.restore(snapshot);
  if (futureIndex > 0) {
    sim.rngState = futureSeed(sim.seed ^ sim.frame, futureIndex);
    sim.director.reshuffleMaterialRemainder();
  }
  const startHeight = sim.height;
  let previous = previousAction;
  let frames = 0;
  for (const action of actions) {
    sim.step(heldActionInput(action, previous));
    previous = action;
    frames++;
    if (sim.dead) break;
  }
  return {
    sim, frames, previous,
    dead: sim.dead,
    gain: sim.height - startHeight,
    survived: !sim.dead,
  };
}

function score(outcomes) {
  // Soft worst-case: dying in some future is heavily penalised but not fatal,
  // otherwise the beam extinguishes as soon as no open-loop option survives
  // every reseeded spawn pattern.
  let total = 0;
  for (const outcome of outcomes) {
    total += outcome.gain - (outcome.dead ? config.deathPenalty : 0);
  }
  const minimum = Math.min(...outcomes.map(
    (o) => o.gain - (o.dead ? config.deathPenalty : 0)));
  return minimum * 2 + total / outcomes.length;
}

const root = new Sim(config.seed, { rules: { autoGuard: false, checkpoints: false } });
let beam = [{
  snapshot: root.snapshot(),
  previousAction: 0,
  actions: [],
  height: root.height,
  frame: 0,
}];

const started = Date.now();
let iteration = 0;
let best = beam[0];

while (best.height < config.target && best.actions.length < config.maxFrames) {
  const candidates = [];
  for (const entry of beam) {
    const probe = new Sim(config.seed, { rules: { autoGuard: false, checkpoints: false } });
    probe.restore(entry.snapshot);
    for (const move of moves(probe)) {
      const actions = expand(move);
      const outcomes = [];
      for (let future = 0; future < config.futures; future++) {
        outcomes.push(play(entry.snapshot, entry.previousAction, actions, config.seed, future));
      }
      const value = score(outcomes);
      const real = outcomes[0];
      if (real.dead) continue;   // cannot continue a branch that is already dead
      candidates.push({
        snapshot: real.sim.snapshot(),
        previousAction: real.previous,
        actions: entry.actions.concat(actions),
        height: real.sim.height,
        frame: real.sim.frame,
        value,
      });
    }
  }
  if (!candidates.length) {
    process.stderr.write('beam extinguished: every option dies in some future\n');
    break;
  }
  candidates.sort((a, b) => b.value - a.value || b.height - a.height);
  beam = candidates.slice(0, config.beam);
  best = beam.reduce((a, b) => (b.height > a.height ? b : a));
  iteration++;
  if (iteration % config.progressEvery === 0) {
    const seconds = (Date.now() - started) / 1000;
    const frames = best.actions.length;
    process.stderr.write(
      `iter ${iteration} frames ${frames} height ${best.height} ` +
      `climb ${(best.height / (frames / 60)).toFixed(1)} h/s ` +
      `elapsed ${seconds.toFixed(0)}s\n`);
  }
}

const verify = play(root.snapshot(), 0, best.actions, config.seed, 0);
const seconds = (Date.now() - started) / 1000;
const summary = {
  seed: config.seed,
  frames: best.actions.length,
  height: verify.sim.height,
  reachedTarget: verify.sim.height >= config.target,
  died: verify.dead,
  climbHeightPerSecond: +(verify.sim.height / (best.actions.length / 60)).toFixed(1),
  beam: config.beam,
  horizon: config.horizon,
  futures: config.futures,
  searchSeconds: +seconds.toFixed(1),
};
process.stderr.write(`${JSON.stringify(summary)}\n`);

if (config.output) {
  const payload = {
    version: 1,
    seed: config.seed,
    frames: best.actions.length,
    targetHeight: verify.sim.height,
    actions: Buffer.from(Uint8Array.from(best.actions)).toString('base64'),
    finalHash: verify.sim.hash(),
    search: summary,
  };
  fs.writeFileSync(config.output, zlib.gzipSync(JSON.stringify(payload)));
  process.stderr.write(`wrote ${config.output}\n`);
}
