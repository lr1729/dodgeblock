#!/usr/bin/env node
// Behavioural audit of a recorded trace: focus economy, dash effectiveness,
// movement, and action mix. Offline and exact — replays through the real sim.
// usage: node rl/audit-trace.mjs trace.json
import fs from 'node:fs';

import { Sim } from '../src/sim/sim.js';
import { heldActionInput } from './env-v2.mjs';

const LANDING_WINDOW = 90;
const ACTION_NAMES = ['neutral', 'left', 'right', 'up', 'up-left', 'up-right', 'down',
  'down-left', 'down-right', 'focus', 'focus-left', 'focus-right', 'focus-up',
  'focus-up-left', 'focus-up-right', 'focus-down', 'focus-down-left', 'focus-down-right'];

const trace = JSON.parse(fs.readFileSync(process.argv[2]));
const seed = (trace.bridge_seed + trace.env_index * 0x9e3779b9 + trace.episode * 0x85ebca6b) >>> 0;
const actions = Buffer.from(trace.actions, 'base64');
const sim = new Sim(seed, { rules: { autoGuard: false, checkpoints: false } });

const events = {};
for (const name of ['focusAimStart', 'focusStart', 'focusKick', 'focusBreak',
  'focusBonk', 'autoGuard']) {
  sim.events.on(name, () => { events[name] = (events[name] ?? 0) + 1; });
}

const dashes = [];
let pending = [], aiming = null;
sim.events.on('focusStart', (event) => {
  aiming = {
    dy: +(event.dy ?? 0).toFixed(2),
    startHeight: sim.height,
    startY: sim.player.y,
    breaks: events.focusBreak ?? 0,
  };
});

const counts = new Array(ACTION_NAMES.length).fill(0);
let previous = 0, wasDashing = false;
let chargeFrames = 0, aimFrames = 0, dashFrames = 0, pressEdges = 0, holdFrames = 0;
let travel = 0, lastX = null;

for (const action of actions) {
  const input = heldActionInput(action, previous);
  if (input.focusPressed) pressEdges++;
  if (input.focusHeld) holdFrames++;
  sim.step(input);
  previous = action;
  counts[action]++;
  if (sim.player.focus > 0) chargeFrames++;
  if (sim.player.focusAimTimer > 0) aimFrames++;
  const dashing = sim.player.focusTimer > 0;
  if (dashing) dashFrames++;
  if (lastX !== null) travel += Math.abs(sim.player.x - lastX);
  lastX = sim.player.x;
  if (wasDashing && !dashing && aiming) {
    aiming.endFrame = sim.frame;
    aiming.rise_px = Math.round(aiming.startY - sim.player.y);
    aiming.breaks = (events.focusBreak ?? 0) - aiming.breaks;
    pending.push(aiming);
    aiming = null;
  }
  wasDashing = dashing;
  for (const entry of pending) {
    if (sim.frame - entry.endFrame >= LANDING_WINDOW) {
      entry.gain = sim.height - entry.startHeight;
      dashes.push(entry);
    }
  }
  pending = pending.filter((entry) => entry.gain === undefined);
  if (sim.dead) break;
}

const frames = sim.frame;
const layers = sim.height / 40;
const meanGain = dashes.length
  ? dashes.reduce((sum, d) => sum + d.gain, 0) / dashes.length : 0;
const baseline = sim.height / (frames / (LANDING_WINDOW + 8));

console.log(JSON.stringify({
  frames,
  seconds: +(frames / 60).toFixed(1),
  height: sim.height,
  deathCause: sim.deathCause,
  climb_h_per_s: +(sim.height / (frames / 60)).toFixed(1),
  auto_guard_fired: events.autoGuard ?? 0,
  focus: {
    charges_spent: events.focusAimStart ?? 0,
    charges_expected: +(3 + layers / 3).toFixed(1),
    press_edges: pressEdges,
    dashes_committed: events.focusStart ?? 0,
    blocks_shattered: events.focusBreak ?? 0,
    kicks: events.focusKick ?? 0,
    bonks: events.focusBonk ?? 0,
    charge_available_fraction: +(chargeFrames / frames).toFixed(4),
    mean_frames_held_before_spending: +(chargeFrames / Math.max(1, events.focusAimStart ?? 1)).toFixed(1),
    frames_in_aim: aimFrames,
    frames_in_dash: dashFrames,
    frames_holding_focus_action: holdFrames,
    charges_at_death: sim.player.focus,
  },
  dash_effectiveness: {
    measured: dashes.length,
    upward: dashes.filter((d) => d.dy < 0).length,
    mean_height_gain_after: +meanGain.toFixed(1),
    baseline_gain_same_window: +baseline.toFixed(1),
    lift_over_baseline: +(meanGain - baseline).toFixed(1),
    zero_gain_dashes: dashes.filter((d) => d.gain === 0).length,
    mean_rise_px: Math.round(dashes.reduce((s, d) => s + d.rise_px, 0) / Math.max(1, dashes.length)),
  },
  movement: {
    horizontal_travel_px: Math.round(travel),
    px_per_second: +(travel / (frames / 60)).toFixed(1),
  },
  action_mix: ACTION_NAMES
    .map((name, index) => [name, +(counts[index] / frames).toFixed(3)])
    .filter(([, fraction]) => fraction >= 0.01)
    .sort((a, b) => b[1] - a[1]),
}, null, 2));
