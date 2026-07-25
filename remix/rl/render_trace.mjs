#!/usr/bin/env node
// Replay a record_trace.py trace through the exact sim and render it to video
// via ffmpeg. Faithful-schematic: real geometry and colors, brightened for
// legibility, plus a phase strip, focus pips, a height bar, and a burned-in
// caption (height / phase / clock) rendered by libass.
// usage: node rl/render_trace.mjs trace.json out.mp4
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
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
const W = GAME_W, H = GAME_H, SCALE = 2;
const frame = Buffer.alloc(W * H * 3);

// Brightened palette: the in-game night sky reads as black once a player
// applies limited-range colour, so the capture uses a lifted daylight ramp.
const SKY_TOP = 0x6f8ab8, SKY_BOTTOM = 0xa9bcd6;
const GRID = 0x8fa4c4, GROUND_FILL = 0x7a5a3c, GROUND_EDGE = 0x9d7850;
const PHASE_COLORS = {
  opening: 0x9aa3b5, calm: 0x46c46a, build: 0xf0bf2a,
  surge: 0xf0453c, release: 0x3f9ae0,
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

function drawSky() {
  for (let y = 0; y < H; y++) {
    const t = y / H;
    const color =
      ((Math.round(((SKY_TOP >> 16) & 255) * (1 - t) + ((SKY_BOTTOM >> 16) & 255) * t) << 16) |
       (Math.round(((SKY_TOP >> 8) & 255) * (1 - t) + ((SKY_BOTTOM >> 8) & 255) * t) << 8) |
        Math.round((SKY_TOP & 255) * (1 - t) + (SKY_BOTTOM & 255) * t));
    fillRect(0, y, W, 1, color);
  }
}

function blockColor(block) {
  if (block.type === 'gravel') return COLOR_GRAVEL;
  if (block.type === 'beam') return COLOR_BEAM;
  return COLOR_BLOCK_FILLS[Math.abs(block.shade ?? 0) % COLOR_BLOCK_FILLS.length];
}

function drawBlock(block, highlight) {
  const y = block.y + sim.camY;
  if (y + block.h < 0 || y > H) return;
  fillRect(block.x, y, block.w, block.h, blockColor(block));
  outlineRect(block.x, y, block.w, block.h, highlight ? 0xffffff : 0x3b2a1c, 2, highlight ? 0.9 : 0.5);
}

function draw() {
  drawSky();
  for (let altitude = 400; altitude <= 12_000; altitude += 400) {
    const y = GROUND.y - altitude + sim.camY;
    if (y > -4 && y < H) fillRect(0, y, W, 1, GRID, 0.45);
  }
  const groundY = GROUND.y + sim.camY;
  if (groundY < H) {
    fillRect(GROUND.x, groundY, GROUND.w, H - groundY, GROUND_FILL);
    fillRect(GROUND.x, groundY, GROUND.w, 3, GROUND_EDGE);
  }
  for (const block of sim.blocks.blocks) drawBlock(block, false);
  for (const block of sim.blocks.falling) drawBlock(block, true);
  for (const forecast of sim.director.forecasts) {
    outlineRect(forecast.x, forecast.y + sim.camY, forecast.w, forecast.h,
      COLOR_WARNING, 3, 0.95);
  }
  const p = sim.player;
  fillRect(p.x, p.y + sim.camY, p.w, p.h, COLOR_PLAYER);
  outlineRect(p.x, p.y + sim.camY, p.w, p.h, 0xffffff, 2, 0.9);
  if (p.focusAimRemaining > 0) {
    outlineRect(p.x - 4, p.y + sim.camY - 4, p.w + 8, p.h + 8, 0x18e0f5, 3);
  } else if (p.focusTimer > 0) {
    outlineRect(p.x - 3, p.y + sim.camY - 3, p.w + 6, p.h + 6, 0xffffff, 3);
  }
  fillRect(0, 0, W, 8, PHASE_COLORS[sim.director.phase] ?? 0x9aa3b5);
  for (let pip = 0; pip < 3; pip++) {
    const charged = p.focus >= pip + 1;
    fillRect(10 + pip * 16, 40, 12, 12, charged ? 0x18e0f5 : 0x33415c, charged ? 1 : 0.7);
    outlineRect(10 + pip * 16, 40, 12, 12, 0x0d2233, 2, 0.8);
  }
  const fraction = Math.min(1, sim.height / 1600);
  fillRect(W - 10, 10, 8, H - 20, 0x33415c, 0.7);
  fillRect(W - 10, 10 + (H - 20) * (1 - fraction), 8, (H - 20) * fraction, 0x2fbf6b);
}

function timecode(centiseconds) {
  const cs = Math.max(0, Math.round(centiseconds));
  const h = Math.floor(cs / 360_000);
  const m = Math.floor(cs / 6000) % 60;
  const s = Math.floor(cs / 100) % 60;
  return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(cs % 100).padStart(2, '0')}`;
}

// One caption per distinct readout so libass stays cheap.
const captions = [];
function captionFor(source, frameIndex) {
  const text =
    `height ${Math.round(source.height)}   phase ${source.director.phase.toUpperCase()}` +
    `   focus ${source.player.focus}/3   ${Math.floor(frameIndex / 60)}s`;
  const previous = captions[captions.length - 1];
  if (previous && previous.text === text) return;
  if (previous) previous.end = frameIndex;
  captions.push({ start: frameIndex, end: frameIndex + 1, text });
}

function writeSubtitles(file) {
  const header = [
    '[Script Info]', 'ScriptType: v4.00+', `PlayResX: ${W * SCALE}`, `PlayResY: ${H * SCALE}`,
    '', '[V4+ Styles]',
    'Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
    'Style: HUD,Noto Sans Mono,30,&H00FFFFFF,&H00101820,&HA0101820,1,3,2,0,7,26,26,26,1',
    '', '[Events]',
    'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
  ];
  const events = captions.map((entry) =>
    `Dialogue: 0,${timecode(entry.start * 100 / 60)},${timecode(entry.end * 100 / 60)},HUD,,0,0,0,,${entry.text}`);
  fs.writeFileSync(file, `${[...header, ...events].join('\n')}\n`);
}

const subtitleFile = path.join(os.tmpdir(), `dodgeblock-hud-${process.pid}.ass`);
const ffmpeg = spawn('ffmpeg', [
  '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${H}`, '-r', '60', '-i', '-',
  '-vf', `scale=${W * SCALE}:${H * SCALE}:flags=neighbor,ass=${subtitleFile}`,
  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
  '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
  '-color_range', 'tv', '-movflags', '+faststart', outFile,
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

// Pass 1: replay on a scratch sim to collect captions — libass needs the
// subtitle file complete before encoding starts.
{
  const probe = new Sim(seed, { rules: { autoGuard: false, checkpoints: false } });
  let previous = 0, index = 0;
  for (const action of actions) {
    probe.step(heldActionInput(action, previous));
    previous = action;
    captionFor(probe, index++);
    if (probe.dead) break;
  }
  captions[captions.length - 1].end = index + 60;
}
writeSubtitles(subtitleFile);

let previous = 0;
for (const action of actions) {
  sim.step(heldActionInput(action, previous));
  previous = action;
  draw();
  await writeFrame();
  if (sim.dead) break;
}
draw();
if (sim.dead) fillRect(0, 0, W, H, 0xd1342f, 0.3);
for (let hold = 0; hold < 60; hold++) await writeFrame();
ffmpeg.stdin.end();
await finished;
fs.unlinkSync(subtitleFile);

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
