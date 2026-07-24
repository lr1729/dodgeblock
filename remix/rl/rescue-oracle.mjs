#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { FOCUS_FRAMES } from '../src/constants.js';
import { Rng } from '../src/sim/rng.js';
import { Sim } from '../src/sim/sim.js';
import { heldActionInput } from './env-v2.mjs';

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

const trials = Math.max(1, Math.floor(numberArg('--trials', 64)));
const horizon = Math.max(1, Math.floor(numberArg('--horizon', 360)));
const futures = Math.max(1, Math.floor(numberArg('--futures', 3)));
const searchSeed = numberArg('--seed', 0x5e5c_0e17) >>> 0;
const output = stringArg('--output', '');
const allCandidates = process.argv.includes('--all-candidates');
const rewindFrames = [...new Set(
  stringArg('--rewinds', '1,30,60,120,240')
    .split(',')
    .map((value) => Math.max(1, Math.floor(Number(value))))
    .filter(Number.isFinite),
)].sort((a, b) => a - b);
if (horizon <= rewindFrames.at(-1)) {
  throw new Error(
    `--horizon (${horizon}) must exceed the largest rewind ` +
    `(${rewindFrames.at(-1)}) so the unchanged fatal trajectory is observed`,
  );
}
const rules = Object.freeze({ autoGuard: false, checkpoints: false });
const rng = new Rng(searchSeed);

function caseFiles(inputs) {
  const result = [];
  for (const input of inputs) {
    const stat = fs.statSync(input);
    if (stat.isDirectory()) {
      for (const filename of fs.readdirSync(input).sort()) {
        if (filename.endsWith('.json.gz')) result.push(path.join(input, filename));
      }
    } else {
      result.push(input);
    }
  }
  return result;
}

function normalOption() {
  const action = rng.weighted([
    { value: 0, w: 0.08 },
    { value: 1, w: 0.08 },
    { value: 2, w: 0.08 },
    { value: 3, w: 0.08 },
    { value: 4, w: 0.22 },
    { value: 5, w: 0.22 },
    { value: 6, w: 0.06 },
    { value: 7, w: 0.09 },
    { value: 8, w: 0.09 },
  ]);
  return Array(rng.int(2, 11)).fill(action);
}

function focusOption() {
  const action = rng.int(9, 18);
  return [
    ...Array(rng.int(1, 13)).fill(action),
    ...Array(FOCUS_FRAMES + 2).fill(action % 9),
  ];
}

function randomActions(deathCase) {
  const actions = [];
  while (actions.length < horizon) {
    const canTryFocus = deathCase.snapshot.player.focus > 0;
    actions.push(...(canTryFocus && rng.chance(0.14) ? focusOption() : normalOption()));
  }
  return actions.slice(0, horizon);
}

function candidateActions(deathCase, trial, branchSuffixes) {
  if (trial === 0) {
    return [
      ...(deathCase.baselineActions ?? [deathCase.fatalAction]),
      ...Array(horizon).fill(deathCase.fatalAction),
    ].slice(0, horizon);
  }
  if (trial <= 54) {
    const zeroBased = trial - 1;
    const action = zeroBased % 18;
    const group = Math.floor(zeroBased / 18);
    branchSuffixes[group] ??= randomActions(deathCase);
    return [
      ...Array(6).fill(action),
      ...branchSuffixes[group].slice(6),
    ];
  }
  return randomActions(deathCase);
}

function futureSeed(deathCase, index) {
  let value = (
    (deathCase.seed >>> 0) ^
    Math.imul((deathCase.frame + 1) >>> 0, 0x9e3779b9) ^
    Math.imul(index >>> 0, 0x85ebca6b) ^
    searchSeed
  ) >>> 0;
  value ^= value >>> 16;
  return Math.imul(value, 0x7feb352d) >>> 0;
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

function rollout(deathCase, actions, futureIndex) {
  const sim = new Sim(deathCase.snapshot.seed, { rules });
  sim.restore(deathCase.snapshot);
  if (futureIndex > 0) {
    sim.seed = futureSeed(deathCase, futureIndex);
    sim.rng.s = sim.seed;
    sim.director.reshuffleMaterialRemainder();
  }
  let previousAction = deathCase.previousAction;
  let frames = 0;
  for (const proposedAction of actions) {
    const action = canonicalAction(sim, proposedAction, previousAction);
    sim.step(heldActionInput(action, previousAction));
    previousAction = action;
    frames++;
    if (sim.dead) break;
  }
  return {
    survived: !sim.dead && frames === actions.length,
    frames,
    height: sim.height,
    focus: sim.player.focus,
  };
}

function rewindView(deathCase, rewind) {
  if (deathCase.version === 1) {
    if (rewind !== 1) return null;
    return { ...deathCase, rewind, baselineActions: [deathCase.fatalAction] };
  }
  if (deathCase.version !== 2) {
    throw new Error(`unsupported death case version: ${deathCase.version}`);
  }
  const actions = [...Buffer.from(deathCase.history.actions, 'base64')];
  if (!actions.length || rewind > actions.length) return null;
  const prefixLength = actions.length - rewind;
  const sim = new Sim(deathCase.history.snapshot.seed, { rules });
  sim.restore(deathCase.history.snapshot);
  let previousAction = deathCase.history.previousAction;
  for (const action of actions.slice(0, prefixLength)) {
    sim.step(heldActionInput(action, previousAction));
    previousAction = action;
    if (sim.dead) {
      throw new Error('captured action history dies before its recorded fatal action');
    }
  }
  return {
    ...deathCase,
    rewind,
    frame: sim.frame,
    snapshot: sim.snapshot(),
    previousAction,
    fatalAction: actions[prefixLength],
    baselineActions: actions.slice(prefixLength),
  };
}

function analyzeView(filename, deathCase) {
  let firstOriginalRescue = null;
  let firstRobustRescue = null;
  let robust = null;
  let best = null;
  const candidates = [];
  const branchSuffixes = [];
  for (let trial = 0; trial < trials; trial++) {
    const actions = candidateActions(deathCase, trial, branchSuffixes);
    const outcomes = Array.from({ length: futures }, (_, index) =>
      rollout(deathCase, actions, index));
    const originalRescue = outcomes[0].survived;
    const robustRescue = outcomes.every((result) => result.survived);
    if (originalRescue && firstOriginalRescue === null) firstOriginalRescue = trial + 1;
    const score = outcomes.reduce(
      (total, result) => total + result.frames + result.height / 100 + result.focus,
      0,
    );
    if (allCandidates) {
      const branchAction = trial > 0 && trial <= 54
        ? canonicalAction(
          deathCase.snapshot,
          (trial - 1) % 18,
          deathCase.previousAction,
        )
        : null;
      candidates.push({
        trial: trial + 1,
        originalRescue,
        robustRescue,
        score,
        branchAction,
        outcomes,
        actions: Buffer.from(actions).toString('base64'),
      });
    }
    if (robustRescue && firstRobustRescue === null) {
      firstRobustRescue = trial + 1;
      robust = {
        score,
        trial: trial + 1,
        outcomes,
        actions: Buffer.from(actions).toString('base64'),
      };
    }
    if (!best || score > best.score) {
      best = {
        score,
        trial: trial + 1,
        originalRescue,
        robustRescue,
        outcomes,
        actions: Buffer.from(actions).toString('base64'),
      };
    }
    if (!allCandidates && firstRobustRescue !== null) break;
  }
  return {
    filename,
    rewind: deathCase.rewind,
    height: deathCase.height,
    cause: deathCase.cause,
    phase: deathCase.phase,
    firstOriginalRescue,
    firstRobustRescue,
    robust,
    best,
    candidates: allCandidates ? candidates : undefined,
  };
}

function analyze(filename) {
  const deathCase = JSON.parse(zlib.gunzipSync(fs.readFileSync(filename)));
  return rewindFrames
    .map((rewind) => rewindView(deathCase, rewind))
    .filter(Boolean)
    .map((view) => analyzeView(filename, view));
}

const inputs = repeatedArgs('--case');
if (!inputs.length) throw new Error('provide at least one --case file or directory');
const files = caseFiles(inputs);
const cases = files.flatMap(analyze);
const budgets = [...new Set([8, 16, 32, trials].filter((value) => value <= trials))]
  .sort((a, b) => a - b);
const payload = {
  version: 2,
  trials,
  horizon,
  futures,
  allCandidates,
  searchSeed,
  rewindFrames,
  caseCount: files.length,
  evaluationCount: cases.length,
  curves: rewindFrames.flatMap((rewind) => {
    const entries = cases.filter((entry) => entry.rewind === rewind);
    return budgets.map((budget) => ({
      rewind,
      budget,
      cases: entries.length,
      originalRescueRate: entries.filter(
        (entry) => entry.firstOriginalRescue !== null &&
          entry.firstOriginalRescue <= budget,
      ).length / Math.max(1, entries.length),
      robustRescueRate: entries.filter(
        (entry) => entry.firstRobustRescue !== null &&
          entry.firstRobustRescue <= budget,
      ).length / Math.max(1, entries.length),
    }));
  }),
  cases,
};
const serialized = `${JSON.stringify(payload, null, 2)}\n`;
if (output) {
  const temporary = `${output}.tmp`;
  fs.writeFileSync(temporary, serialized);
  fs.renameSync(temporary, output);
} else {
  process.stdout.write(serialized);
}
