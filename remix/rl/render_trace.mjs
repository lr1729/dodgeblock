#!/usr/bin/env node
// Replay a record_trace.py trace through the exact sim and render it to video
// via ffmpeg (rawvideo pipe). Faithful-schematic: real geometry and colors,
// plus a phase strip (top), focus pips (top-left), and a height bar (right).
// usage: node rl/render_trace.mjs trace.json out.mp4
import fs from 'node:fs';
import { spawn } from 'node:child_process';

import { Sim } from '../src/sim/sim.js';
import { heldActionInput } from './env-v2.mjs';
import {
  GAME_W, GAME_H, GROUND,
  COLOR_BLOCK_FILLS, COLOR_GRAVEL, COLOR_BEAM,
  COLOR_PLAYER, COLOR_WARNING,
} from '../src/constants.js';

const [traceFile, outFile] = process.argv.slice(2);
if (!traceFile || !outFile) throw new Error('usage: render_trace.mjs trace.json out.mp4');
const trace = JSON.parse(fs.readFileSync(traceFile));
const seed = (trace.bridge_seed + trace.env_index * 0x9e3779b9 + trace.episode * 0x85ebca6b) >>> 0;
const actions = Buffer.from(trace.actions, 'base64');
if (actions.length !== trace.frames) throw new Error('action length mismatch');

const sim = new Sim(seed, { rules: { autoGuard: false, checkpoints: false } });
const W = GAME_W, H = GAME_H;
const frame = Buffer.alloc(W * H * 3);

const PHASE_COLORS = {
  opening: 0x5a5a66, calm: 0x3fa34d, build: 0xd9a521,
  surge: 0xd1342f, release: 0x3b82c4,
};

function fillRect(x, y, w, h, color, alpha = 1) {
  const r = (color >> 16) & 255, g = (color >> 8) & 255, b = color & 255;
  const x0 = Math.max(0, Math.round(x)), x1 = Math.min(W, Math.round(x + w));
  const y0 = Math.max(0, Math.round(y)), y1 = Math.min(H, Math.round(y + h));
  for (let yy = y0; yy < y1; yy++) {
    let offset = (yy * W + x0) * 3;
    for (let xx = x0; xx < x1; xx++, offset += 3) {
      frame[offset] = frame[offset] * (1 - alpha) + r * alpha;
      frame[offset + 1] = frame[offset + 1] * (1 - alpha) + g * alpha;
      frame[offset + 2] = frame[offset + 2] * (1 - alpha) + b * alpha;
    }
  }
}

function outlineRect(x, y, w, h, color, thickness = 2, alpha = 1) {
  fillRect(x, y, w, thickness, color, alpha);
  fillRect(x, y + h - thickness, w, thickness, color, alpha);
  fillRect(x, y, thickness, h, color, alpha);
  fillRect(x + w - thickness, y, thickness, h, color, alpha);
}

function blockColor(block) {
  if (block.type === 'gravel') return COLOR_GRAVEL;
  if (block.type === 'beam') return COLOR_BEAM;
  return COLOR_BLOCK_FILLS[Math.abs(block.shade ?? 0) % COLOR_BLOCK_FILLS.length];
}

function draw() {
  fillRect(0, 0, W, H, 0x181c2c);
  // altitude gridlines every 400 world px (10 layers)
  for (let altitude = 400; altitude <= 12_000; altitude += 400) {
    const y = GROUND.y - altitude + sim.camY;
    if (y > -4 && y < H) fillRect(0, y, W, 1, 0x2c3350);
  }
  const groundY = GROUND.y + sim.camY;
  if (groundY < H) fillRect(GROUND.x, groundY, GROUND.w, H - groundY, 0x3a2d22);
  for (const block of sim.blocks.blocks) {
    const y = block.y + sim.camY;
    if (y + block.h < 0 || y > H) continue;
    fillRect(block.x, y, block.w, block.h, blockColor(block));
    outlineRect(block.x, y, block.w, block.h, 0x000000, 1, 0.35);
  }
  for (const block of sim.blocks.falling) {
    const y = block.y + sim.camY;
    if (y + block.h < 0 || y > H) continue;
    fillRect(block.x, y, block.w, block.h, blockColor(block));
    outlineRect(block.x, y, block.w, block.h, 0xffffff, 1, 0.5);
  }
  for (const forecast of sim.director.forecasts) {
    outlineRect(forecast.x, forecast.y + sim.camY, forecast.w, forecast.h,
      COLOR_WARNING, 2, 0.85);
  }
  const p = sim.player;
  fillRect(p.x, p.y + sim.camY, p.w, p.h, COLOR_PLAYER);
  if (p.focusAimRemaining > 0) {
    outlineRect(p.x - 3, p.y + sim.camY - 3, p.w + 6, p.h + 6, 0x39d5e8, 2);
  } else if (p.focusTimer > 0) {
    outlineRect(p.x - 2, p.y + sim.camY - 2, p.w + 4, p.h + 4, 0xffffff, 2);
  }
  // HUD: phase strip, focus pips, height bar (fraction of 1k per bar segment)
  fillRect(0, 0, W, 6, PHASE_COLORS[sim.director.phase] ?? 0x5a5a66);
  for (let pip = 0; pip < 3; pip++) {
    const charged = sim.player.focus >= pip + 1;
    fillRect(8 + pip * 14, 12, 10, 10, charged ? 0x39d5e8 : 0x2c3350);
    outlineRect(8 + pip * 14, 12, 10, 10, 0x39d5e8, 1, 0.6);
  }
  const fraction = Math.min(1, sim.height / 1000);
  fillRect(W - 8, 6, 6, H - 6, 0x2c3350, 0.8);
  fillRect(W - 8, 6 + (H - 6) * (1 - fraction), 6, (H - 6) * fraction, 0x74d99a);
}

const ffmpeg = spawn('ffmpeg', [
  '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${H}`, '-r', '60',
  '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', outFile,
], { stdio: ['pipe', 'ignore', 'inherit'] });
const finished = new Promise((resolve, reject) => {
  ffmpeg.on('close', (code) => code === 0 ? resolve() : reject(new Error(`ffmpeg ${code}`)));
});

function writeFrame() {
  return new Promise((resolve) => {
    if (ffmpeg.stdin.write(Buffer.from(frame))) resolve();
    else ffmpeg.stdin.once('drain', resolve);
  });
}

let previous = 0;
for (const action of actions) {
  sim.step(heldActionInput(action, previous));
  previous = action;
  draw();
  await writeFrame();
  if (sim.dead) break;
}
draw();
if (sim.dead) fillRect(0, 0, W, H, 0xd1342f, 0.25);
for (let hold = 0; hold < 60; hold++) await writeFrame();
ffmpeg.stdin.end();
await finished;

const verdict = {
  replayHeight: sim.height,
  traceHeight: trace.height,
  match: Math.abs(sim.height - trace.height) < 1e-6,
  dead: sim.dead,
  frames: actions.length,
  seconds: Math.round(actions.length / 60),
};
console.log(JSON.stringify(verdict));
if (!verdict.match) process.exit(1);
