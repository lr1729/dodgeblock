import {
  ARENA_COLS,
  ARENA_W,
  ARENA_X,
  BLOCK_FALL_CAP,
  BLOCK_H,
  CAMERA_ANCHOR_Y,
  FOCUS_AIM_MAX_FRAMES,
  FOCUS_CAP,
  FOCUS_FRAMES,
  FOCUS_RECHARGE_LAYERS,
  GAME_H,
  GROUND,
  MOVE_SPEED,
  PLAYER_FALL_CAP,
  SPAWN_GRID,
  TELEGRAPH_FRAMES,
} from '../src/constants.js';
import { framesToY } from '../src/sim/director.js';

export const ACTION_COUNT = 18;
export const TERRAIN_ROWS = 32;
export const TERRAIN_COLS = ARENA_COLS;
export const TERRAIN_SIZE = TERRAIN_ROWS * TERRAIN_COLS;
export const SKYLINE_SIZE = ARENA_COLS;
export const FALLING_COUNT = 32;
export const FALLING_FEATURES = 11;
export const FORECAST_COUNT = 16;
export const FORECAST_FEATURES = 10;
export const STATE_SIZE = 32;

const TERRAIN_ABOVE_ROWS = 13;
const MATERIAL_CODE = Object.freeze({ wood: 1, gravel: 2, beam: 3 });
const MATERIAL_VECTOR = Object.freeze({
  wood: [1, 0, 0],
  gravel: [0, 1, 0],
  beam: [0, 0, 1],
});

export function decodeAction(action) {
  const bounded = Math.max(0, Math.min(ACTION_COUNT - 1, Number(action) || 0));
  const focus = bounded >= 9;
  const local = bounded % 9;
  const verticalIndex = Math.floor(local / 3);
  const horizontalIndex = local % 3;
  return {
    focus,
    horizontal: horizontalIndex === 1 ? -1 : horizontalIndex === 2 ? 1 : 0,
    vertical: verticalIndex === 1 ? -1 : verticalIndex === 2 ? 1 : 0,
  };
}

export function heldActionInput(action, previousAction = 0) {
  const held = decodeAction(action);
  const previous = decodeAction(previousAction);
  const up = held.vertical < 0;
  return {
    up,
    down: held.vertical > 0,
    left: held.horizontal < 0,
    right: held.horizontal > 0,
    jumpPressed: up && previous.vertical >= 0,
    focusPressed: held.focus && !previous.focus,
    focusReleased: !held.focus && previous.focus,
    focusHeld: held.focus,
    focusDirX: 0,
    focusDirY: 0,
  };
}

function materialVector(type) {
  return MATERIAL_VECTOR[type] ?? [0, 0, 0];
}

function terrainViewTop(sim) {
  const playerScreenY = sim.player.y + sim.camY;
  const playerRow = Math.floor(playerScreenY / BLOCK_H);
  return (playerRow - TERRAIN_ABOVE_ROWS) * BLOCK_H - sim.camY;
}

function fillTerrain(terrain, block, viewTop, flags) {
  const col0 = Math.max(0, Math.floor((block.x - ARENA_X) / SPAWN_GRID));
  const col1 = Math.min(
    TERRAIN_COLS - 1,
    Math.ceil((block.x + block.w - ARENA_X) / SPAWN_GRID) - 1,
  );
  const row0 = Math.max(0, Math.floor((block.y - viewTop) / BLOCK_H));
  const row1 = Math.min(
    TERRAIN_ROWS - 1,
    Math.ceil((block.y + block.h - viewTop) / BLOCK_H) - 1,
  );
  if (col1 < col0 || row1 < row0) return;
  for (let row = row0; row <= row1; row++) {
    for (let col = col0; col <= col1; col++) {
      const index = row * TERRAIN_COLS + col;
      terrain[index] |= flags;
    }
  }
}

function focusPreview(sim) {
  if (sim.player.focusAimTimer <= 0) return { target: null, branch: new Set() };
  const collision = sim.focusPathCollision();
  if (collision.kind !== 'fixed') return { target: collision.block ?? null, branch: new Set() };
  return {
    target: collision.block,
    branch: new Set(sim.blocks.dependentBranch(collision.block)),
  };
}

function encodeTerrain(sim, terrain) {
  terrain.fill(0);
  const viewTop = terrainViewTop(sim);
  const preview = focusPreview(sim);
  for (const block of sim.blocks.blocks) {
    if (!block.fixed) continue;
    let flags = MATERIAL_CODE[block.type] ?? 0;
    if (block.faultTimer > 0) flags |= 1 << 2;
    if (preview.branch.has(block)) flags |= 1 << 3;
    if (preview.target === block) flags |= 1 << 4;
    fillTerrain(terrain, block, viewTop, flags);
  }
  fillTerrain(terrain, sim.player, viewTop, 1 << 5);
  return viewTop;
}

function encodeSkyline(sim, skyline) {
  const playerLayer = Math.floor((GROUND.y - (sim.player.y + sim.player.h)) / BLOCK_H);
  const surface = sim.blocks.surfaceLayers();
  for (let col = 0; col < SKYLINE_SIZE; col++) {
    const relative = Math.max(-127, Math.min(127, (surface[col] ?? 0) - playerLayer));
    skyline[col] = relative + 128;
  }
}

function encodeFalling(sim, output) {
  output.fill(0);
  const p = sim.player;
  const falling = [...sim.blocks.falling]
    .sort((a, b) => {
      const aEta = framesToY(a, p.y, 600);
      const bEta = framesToY(b, p.y, 600);
      return aEta - bEta || Math.abs(a.x - p.x) - Math.abs(b.x - p.x) || a.id - b.id;
    })
    .slice(0, FALLING_COUNT);
  for (let index = 0; index < falling.length; index++) {
    const block = falling[index];
    const [wood, gravel, beam] = materialVector(block.type);
    const offset = index * FALLING_FEATURES;
    output.set([
      (block.x + block.w / 2 - (p.x + p.w / 2)) / ARENA_W,
      (block.y + block.h / 2 - (p.y + p.h / 2)) / GAME_H,
      block.w / ARENA_W,
      block.h / BLOCK_H,
      block.yVel / BLOCK_FALL_CAP,
      wood,
      gravel,
      beam,
      block.spec.lethal ? 1 : 0,
      block.fixedAtFrame === sim.frame ? 1 : 0,
      1,
    ], offset);
  }
  return Math.max(0, sim.blocks.falling.length - FALLING_COUNT);
}

function encodeForecasts(sim, output) {
  output.fill(0);
  const p = sim.player;
  const forecasts = [...sim.director.forecasts]
    .sort((a, b) => {
      const aEta = a.frames + framesToY({ ...a, yVel: 0 }, p.y, 600);
      const bEta = b.frames + framesToY({ ...b, yVel: 0 }, p.y, 600);
      return aEta - bEta || Math.abs(a.x - p.x) - Math.abs(b.x - p.x);
    })
    .slice(0, FORECAST_COUNT);
  for (let index = 0; index < forecasts.length; index++) {
    const forecast = forecasts[index];
    const [wood, gravel, beam] = materialVector(forecast.type);
    const eta = forecast.frames + framesToY({ ...forecast, yVel: 0 }, p.y, 600);
    const offset = index * FORECAST_FEATURES;
    output.set([
      (forecast.x + forecast.w / 2 - (p.x + p.w / 2)) / ARENA_W,
      (forecast.y + forecast.h / 2 + sim.camY) / GAME_H,
      forecast.w / ARENA_W,
      forecast.h / BLOCK_H,
      forecast.frames / TELEGRAPH_FRAMES,
      Number.isFinite(eta) ? Math.min(1, eta / 600) : 1,
      wood,
      gravel,
      beam,
      1,
    ], offset);
  }
  return Math.max(0, sim.director.forecasts.length - FORECAST_COUNT);
}

function phaseVector(phase) {
  return ['opening', 'calm', 'build', 'surge', 'release'].map((id) => phase === id ? 1 : 0);
}

function encodeState(sim, state, controller, overflow) {
  const p = sim.player;
  const held = decodeAction(controller.previousAction);
  const stableLayer = Math.max(0, (GROUND.y - (p.y + p.h)) / BLOCK_H);
  state.set([
    (p.x + p.w / 2 - ARENA_X) / ARENA_W,
    (p.y + sim.camY - CAMERA_ANCHOR_Y) / GAME_H,
    p.xVel / MOVE_SPEED,
    p.yVel / PLAYER_FALL_CAP,
    p.offGround === 0 ? 1 : 0,
    Math.min(1, p.offGround / 10),
    p.wallSide,
    p.wallCoyoteSide,
    Math.min(1, p.wallCoyoteFrames / 6),
    p.lastWallJumpSide,
    p.focus / FOCUS_CAP,
    p.focusProgress / FOCUS_RECHARGE_LAYERS,
    p.focusAimTimer / FOCUS_AIM_MAX_FRAMES,
    p.focusAimRemaining / FOCUS_AIM_MAX_FRAMES,
    p.focusTimer / FOCUS_FRAMES,
    p.focusDX,
    p.focusDY,
    held.horizontal,
    held.vertical,
    held.focus ? 1 : 0,
    Math.log1p(sim.height) / Math.log1p(10_000),
    Math.log1p(Math.max(0, stableLayer)) / Math.log1p(250),
    Math.min(2, (sim.blockRate ?? 0) / 7),
    sim.director.pressure ?? 0,
    Math.min(2, sim.director.elapsedFrames / (60 * 60 * 20)),
    ...phaseVector(sim.director.phase),
    Math.min(1, overflow.falling / FALLING_COUNT),
    Math.min(1, overflow.forecasts / FORECAST_COUNT),
  ]);
}

export function createObservation() {
  return {
    terrain: new Uint8Array(TERRAIN_SIZE),
    skyline: new Uint8Array(SKYLINE_SIZE),
    falling: new Float32Array(FALLING_COUNT * FALLING_FEATURES),
    forecasts: new Float32Array(FORECAST_COUNT * FORECAST_FEATURES),
    state: new Float32Array(STATE_SIZE),
  };
}

export function encodeObservation(sim, controller, observation = createObservation()) {
  encodeTerrain(sim, observation.terrain);
  encodeSkyline(sim, observation.skyline);
  const overflow = {
    falling: encodeFalling(sim, observation.falling),
    forecasts: encodeForecasts(sim, observation.forecasts),
  };
  encodeState(sim, observation.state, controller, overflow);
  return observation;
}

export function observationByteSize() {
  return TERRAIN_SIZE + SKYLINE_SIZE +
    (FALLING_COUNT * FALLING_FEATURES + FORECAST_COUNT * FORECAST_FEATURES + STATE_SIZE) * 4;
}
