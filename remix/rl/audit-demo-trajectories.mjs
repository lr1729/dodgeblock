#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { Sim } from '../src/sim/sim.js';
import { heldActionInput } from './env-v2.mjs';

const filenames = process.argv.slice(2).map((filename) => path.resolve(filename));
if (!filenames.length) {
  throw new Error('usage: node rl/audit-demo-trajectories.mjs DEMO.json.gz...');
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

function increment(object, key, amount = 1) {
  object[key] = (object[key] ?? 0) + amount;
}

function audit(filename) {
  const payload = JSON.parse(zlib.gunzipSync(fs.readFileSync(filename)));
  const actions = Buffer.from(payload.actions, 'base64');
  const sim = new Sim(payload.seed, {
    rules: Object.freeze({ autoGuard: false, checkpoints: false }),
  });
  let previousAction = 0;
  const bands = {};
  const focus = {
    presses: 0,
    commits: 0,
    collisionKinds: {},
    fixedBranchBlocks: 0,
    commitHeights: [],
  };
  for (const action of actions) {
    const band = Math.floor(sim.height / 400) * 400;
    const stats = bands[band] ?? {
      frames: 0,
      sheltered: 0,
      reliefLayers: 0,
    };
    const surface = sim.blocks.surfaceLayers();
    stats.frames++;
    stats.sheltered += Number(isSheltered(sim));
    stats.reliefLayers += Math.max(...surface) - Math.min(...surface);
    bands[band] = stats;

    if (action >= 9 && previousAction < 9) focus.presses++;
    const beforeAim = sim.player.focusAimTimer;
    const collision = beforeAim > 0 ? sim.focusPathCollision() : null;
    sim.step(heldActionInput(action, previousAction));
    if (
      beforeAim > 0 &&
      sim.player.focusAimTimer <= 0 &&
      sim.player.focusTimer > 0
    ) {
      focus.commits++;
      const kind = collision?.kind ?? 'none';
      increment(focus.collisionKinds, kind);
      if (kind === 'fixed') {
        focus.fixedBranchBlocks += sim.blocks.dependentBranch(collision.block).length;
      }
      focus.commitHeights.push(sim.height);
    }
    previousAction = action;
    if (sim.dead) throw new Error(`${filename} dies at frame ${sim.frame}`);
  }
  if (sim.hash() !== payload.finalHash) {
    throw new Error(`${filename} final hash mismatch`);
  }
  return {
    seed: payload.seed,
    frames: actions.length,
    height: sim.height,
    focus,
    bands: Object.entries(bands).map(([height, stats]) => ({
      height: Number(height),
      frames: stats.frames,
      shelteredFraction: stats.sheltered / stats.frames,
      meanReliefLayers: stats.reliefLayers / stats.frames,
    })),
  };
}

const trajectories = filenames.map(audit);
const payload = {
  version: 1,
  trajectories,
  totals: {
    frames: trajectories.reduce((sum, item) => sum + item.frames, 0),
    focusPresses: trajectories.reduce((sum, item) => sum + item.focus.presses, 0),
    focusCommits: trajectories.reduce((sum, item) => sum + item.focus.commits, 0),
  },
};
process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
