#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import {
  ARENA_X,
  BLOCK_H,
  FOCUS_FRAMES,
  GROUND,
  SPAWN_GRID,
} from '../src/constants.js';
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

const seed = numberArg('--seed', 7) >>> 0;
const searchSeed = numberArg('--search-seed', seed ^ 0x6f2d4b19) >>> 0;
const targetHeight = Math.max(BLOCK_H, numberArg('--target-height', 10_000));
const iterations = Math.max(1, Math.floor(numberArg('--iterations', 2_000_000)));
const exploreFrames = Math.max(1, Math.floor(numberArg('--explore-frames', 240)));
const archiveCapacity = Math.max(128, Math.floor(numberArg('--archive-capacity', 2048)));
const checkpointInterval = Math.max(1, Math.floor(numberArg('--checkpoint-interval', 50_000)));
const outputDir = path.resolve(stringArg('--output-dir', 'rl/go-explore-output'));
const resumePath = stringArg('--resume', '');
const rules = Object.freeze({ autoGuard: false, checkpoints: false });
const rng = new Rng(searchSeed);

let nextEntryId = 1;
let bestHeight = 0;
let additions = 0;
const archive = new Map();
let startIteration = 1;

function stable(sim) {
  const player = sim.player;
  return player.offGround === 0 &&
    player.focusAimTimer <= 0 &&
    player.focusTimer <= 0 &&
    (!player.supportBlock ||
      (player.supportBlock.fixed && player.supportBlock.faultTimer <= 0));
}

function surfaceSignature(sim) {
  const playerLayer = Math.round((GROUND.y - (sim.player.y + sim.player.h)) / BLOCK_H);
  const surface = sim.blocks.surfaceLayers();
  const signature = [];
  for (let start = 0; start < surface.length; start += 4) {
    const local = surface.slice(start, start + 4);
    const relative = Math.max(...local) - playerLayer;
    signature.push(Math.max(-6, Math.min(10, relative)));
  }
  return signature.join(',');
}

function cellKey(sim) {
  const playerColumn = Math.floor((sim.player.x - ARENA_X) / (SPAWN_GRID * 2));
  const heightBucket = Math.floor(sim.height / BLOCK_H);
  const phase = sim.director.phase;
  return [
    heightBucket,
    playerColumn,
    sim.player.focus,
    Math.floor(sim.player.focusProgress),
    phase,
    surfaceSignature(sim),
  ].join('|');
}

function entryQuality(entry) {
  return entry.focus * 1000 - entry.frame - entry.trajectory.actions.length * 0.1;
}

function addCell(sim, previousAction, parent, actions, force = false) {
  if (!force && !stable(sim)) return null;
  const key = cellKey(sim);
  const candidate = {
    id: nextEntryId++,
    key,
    snapshot: sim.snapshot(),
    previousAction,
    trajectory: {
      parent: parent?.trajectory ?? null,
      actions: Uint8Array.from(actions),
    },
    height: sim.height,
    frame: sim.frame,
    focus: sim.player.focus,
    chosen: 0,
    children: 0,
  };
  const incumbent = archive.get(key);
  if (incumbent && entryQuality(incumbent) >= entryQuality(candidate)) return incumbent;
  archive.set(key, candidate);
  if (parent) parent.children++;
  additions++;
  bestHeight = Math.max(bestHeight, candidate.height);
  pruneArchive();
  return candidate;
}

function pruneArchive() {
  if (archive.size <= archiveCapacity) return;
  const bandCounts = new Map();
  for (const entry of archive.values()) {
    const band = Math.floor(entry.height / (BLOCK_H * 5));
    bandCounts.set(band, (bandCounts.get(band) ?? 0) + 1);
  }
  const removable = [...archive.values()]
    .filter((entry) => {
      const band = Math.floor(entry.height / (BLOCK_H * 5));
      return entry.trajectory.parent !== null && (bandCounts.get(band) ?? 0) > 1;
    })
    .sort((a, b) => {
      const aFrontier = a.height >= bestHeight - BLOCK_H * 8 ? 1 : 0;
      const bFrontier = b.height >= bestHeight - BLOCK_H * 8 ? 1 : 0;
      if (aFrontier !== bFrontier) return aFrontier - bFrontier;
      const aScore = a.height - Math.log1p(a.chosen) * BLOCK_H * 2 +
        Math.min(4, a.children) * BLOCK_H;
      const bScore = b.height - Math.log1p(b.chosen) * BLOCK_H * 2 +
        Math.min(4, b.children) * BLOCK_H;
      return aScore - bScore;
    });
  const removeCount = archive.size - archiveCapacity;
  for (let index = 0; index < removeCount && index < removable.length; index++) {
    const entry = removable[index];
    const band = Math.floor(entry.height / (BLOCK_H * 5));
    if ((bandCounts.get(band) ?? 0) <= 1) continue;
    archive.delete(entry.key);
    bandCounts.set(band, bandCounts.get(band) - 1);
  }
}

function selectCell() {
  const weighted = [];
  for (const entry of archive.values()) {
    const gap = Math.max(0, bestHeight - entry.height);
    const frontier = Math.exp(-gap / (BLOCK_H * 6));
    const underexplored = 1 / Math.sqrt(entry.chosen + 1);
    const productive = 1 + Math.min(4, entry.children);
    weighted.push({
      value: entry,
      w: 0.01 + frontier * underexplored * productive,
    });
  }
  const selected = rng.weighted(weighted);
  selected.chosen++;
  return selected;
}

function normalOption() {
  const action = rng.weighted([
    { value: 0, w: 0.06 },
    { value: 1, w: 0.10 },
    { value: 2, w: 0.10 },
    { value: 3, w: 0.10 },
    { value: 4, w: 0.32 },
    { value: 5, w: 0.32 },
  ]);
  const duration = rng.int(2, 11);
  return Array(duration).fill(action);
}

function focusOption() {
  const direction = rng.weighted([
    { value: 10, w: 0.10 },
    { value: 11, w: 0.10 },
    { value: 12, w: 0.28 },
    { value: 13, w: 0.20 },
    { value: 14, w: 0.20 },
    { value: 15, w: 0.04 },
    { value: 16, w: 0.04 },
    { value: 17, w: 0.04 },
  ]);
  const aimFrames = rng.int(1, 13);
  return [
    ...Array(aimFrames).fill(direction),
    ...Array(FOCUS_FRAMES + 2).fill(direction % 9),
  ];
}

function nextOption(sim) {
  const player = sim.player;
  const canFocus = player.focus > 0 &&
    player.focusAimTimer <= 0 &&
    player.focusTimer <= 0;
  return canFocus && rng.chance(0.08) ? focusOption() : normalOption();
}

function reconstructActions(entry, tail = []) {
  const segments = [Uint8Array.from(tail)];
  for (let cursor = entry?.trajectory; cursor; cursor = cursor.parent) {
    segments.push(cursor.actions);
  }
  segments.reverse();
  const size = segments.reduce((total, segment) => total + segment.length, 0);
  const actions = Buffer.allocUnsafe(size);
  let offset = 0;
  for (const segment of segments) {
    Buffer.from(segment).copy(actions, offset);
    offset += segment.length;
  }
  return actions;
}

function writeDemonstration(entry, tail, sim, iteration) {
  const actions = reconstructActions(entry, tail);
  const payload = {
    version: 1,
    seed,
    searchSeed,
    targetHeight,
    height: sim.height,
    frames: actions.length,
    iteration,
    finalHash: sim.hash(),
    actions: actions.toString('base64'),
  };
  const filename = `demo-${seed}-${Math.round(sim.height)}-${iteration}.json.gz`;
  const target = path.join(outputDir, filename);
  const temporary = `${target}.tmp`;
  fs.writeFileSync(temporary, zlib.gzipSync(JSON.stringify(payload)));
  fs.renameSync(temporary, target);
  return target;
}

function writeStatus(iteration, started) {
  const frontier = [...archive.values()]
    .sort((a, b) => b.height - a.height || a.frame - b.frame)
    .slice(0, 32)
    .map((entry) => ({
      height: entry.height,
      frame: entry.frame,
      focus: entry.focus,
      chosen: entry.chosen,
      children: entry.children,
    }));
  const payload = {
    version: 1,
    seed,
    searchSeed,
    targetHeight,
    iteration,
    bestHeight,
    archiveSize: archive.size,
    additions,
    elapsedSeconds: Math.round((Date.now() - started) / 100) / 10,
    frontier,
  };
  const target = path.join(outputDir, 'status.json');
  const temporary = `${target}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(payload, null, 2));
  fs.renameSync(temporary, target);
  process.stdout.write(`${JSON.stringify({ event: 'progress', ...payload })}\n`);
  writeSearchCheckpoint(iteration);
}

function writeSearchCheckpoint(iteration) {
  const bands = new Map();
  for (const entry of archive.values()) {
    const band = Math.floor(entry.height / (BLOCK_H * 5));
    if (!bands.has(band)) bands.set(band, []);
    bands.get(band).push(entry);
  }
  for (const entries of bands.values()) {
    entries.sort((a, b) =>
      entryQuality(b) - entryQuality(a) ||
      b.height - a.height ||
      a.chosen - b.chosen);
  }
  const selected = [];
  const bandIds = [...bands.keys()].sort((a, b) => b - a);
  for (let rank = 0; selected.length < 512; rank++) {
    let added = false;
    for (const band of bandIds) {
      const entry = bands.get(band)[rank];
      if (!entry) continue;
      selected.push(entry);
      added = true;
      if (selected.length >= 512) break;
    }
    if (!added) break;
  }
  const entries = selected
    .map((entry) => ({
      key: entry.key,
      snapshot: entry.snapshot,
      previousAction: entry.previousAction,
      actions: reconstructActions(entry).toString('base64'),
      height: entry.height,
      frame: entry.frame,
      focus: entry.focus,
      chosen: entry.chosen,
      children: entry.children,
    }));
  const payload = {
    version: 1,
    seed,
    searchSeed,
    targetHeight,
    iteration,
    rngState: rng.s,
    bestHeight,
    additions,
    entries,
  };
  const target = path.join(outputDir, 'search-checkpoint.json.gz');
  const temporary = `${target}.tmp`;
  fs.writeFileSync(temporary, zlib.gzipSync(JSON.stringify(payload)));
  fs.renameSync(temporary, target);
}

fs.mkdirSync(outputDir, { recursive: true });
const initial = new Sim(seed, { rules });
const root = addCell(initial, 0, null, [], true);
if (resumePath) {
  const payload = JSON.parse(zlib.gunzipSync(fs.readFileSync(resumePath)));
  if (
    payload.version !== 1 ||
    payload.seed !== seed ||
    payload.searchSeed !== searchSeed ||
    payload.targetHeight !== targetHeight
  ) {
    throw new Error('search checkpoint does not match seed, search seed, or target');
  }
  archive.clear();
  archive.set(root.key, root);
  nextEntryId = 2;
  for (const saved of payload.entries) {
    const actions = Buffer.from(saved.actions, 'base64');
    const entry = {
      id: nextEntryId++,
      key: saved.key,
      snapshot: saved.snapshot,
      previousAction: saved.previousAction,
      trajectory: {
        parent: root.trajectory,
        actions: Uint8Array.from(actions),
      },
      height: saved.height,
      frame: saved.frame,
      focus: saved.focus,
      chosen: saved.chosen,
      children: saved.children,
    };
    archive.set(entry.key, entry);
  }
  rng.s = payload.rngState >>> 0;
  bestHeight = payload.bestHeight;
  additions = payload.additions;
  startIteration = payload.iteration + 1;
}
const started = Date.now();

for (let iteration = startIteration; iteration <= iterations; iteration++) {
  const origin = selectCell();
  const sim = new Sim(seed, { rules });
  sim.restore(origin.snapshot);
  let previousAction = origin.previousAction;
  const actions = [];
  let option = [];
  let nextArchiveHeight = Math.max(origin.height + BLOCK_H, bestHeight + BLOCK_H);

  for (let frame = 0; frame < exploreFrames && !sim.dead; frame++) {
    if (!option.length) option = nextOption(sim);
    const action = option.shift();
    sim.step(heldActionInput(action, previousAction));
    previousAction = action;
    actions.push(action);

    if (!sim.dead && sim.height >= targetHeight) {
      const demonstration = writeDemonstration(origin, actions, sim, iteration);
      bestHeight = Math.max(bestHeight, sim.height);
      process.stdout.write(`${JSON.stringify({
        event: 'success',
        iteration,
        height: sim.height,
        frames: reconstructActions(origin, actions).length,
        demonstration,
      })}\n`);
      writeStatus(iteration, started);
      process.exit(0);
    }

    if (sim.height >= nextArchiveHeight && stable(sim)) {
      addCell(sim, previousAction, origin, actions);
      nextArchiveHeight = sim.height + BLOCK_H;
    } else if (stable(sim) && frame > 0 && frame % 30 === 0) {
      addCell(sim, previousAction, origin, actions);
    }
  }

  if (iteration % checkpointInterval === 0) writeStatus(iteration, started);
}

writeStatus(iterations, started);
process.exitCode = 2;
