import {
  ARENA_X,
  ARENA_W,
  BLOCK_H,
  GAME_H,
  MOVE_SPEED,
  PLAYER_FALL_CAP,
  SPAWN_GRID,
} from '../src/constants.js';

export const OBS_CHANNELS = 8;
export const OBS_ROWS = 20;
export const OBS_COLS = Math.round(ARENA_W / SPAWN_GRID);
export const OBS_SIZE = OBS_CHANNELS * OBS_ROWS * OBS_COLS;
export const STATE_SIZE = 16;
export const ACTION_COUNT = 14;
const VIEW_TOP = -280;

const FOCUS_DIRECTIONS = [
  [-1, 0], [1, 0], [0, -1], [0, 1],
  [-1, -1], [1, -1], [-1, 1], [1, 1],
];

export function actionInput(action, wasFocusHeld = false, wasJumpHeld = false) {
  const input = {
    up: false, down: false, left: false, right: false,
    jumpPressed: false, focusPressed: false, focusReleased: false,
    focusHeld: false, focusDirX: 0, focusDirY: 0,
  };
  if (action >= 6 && action < ACTION_COUNT) {
    const [x, y] = FOCUS_DIRECTIONS[action - 6];
    input.focusHeld = true;
    input.focusPressed = !wasFocusHeld;
    input.focusDirX = x;
    input.focusDirY = y;
    return { input, focusHeld: true, jumpHeld: false };
  }

  input.focusReleased = wasFocusHeld;
  input.left = action === 1 || action === 4;
  input.right = action === 2 || action === 5;
  input.up = action >= 3 && action <= 5;
  input.jumpPressed = input.up && !wasJumpHeld;
  return { input, focusHeld: false, jumpHeld: input.up };
}

function fillRect(grid, channel, x, y, w, h, value) {
  const col0 = Math.max(0, Math.floor((x - ARENA_X) / SPAWN_GRID));
  const col1 = Math.min(OBS_COLS - 1, Math.ceil((x + w - ARENA_X) / SPAWN_GRID) - 1);
  const row0 = Math.max(0, Math.floor((y - VIEW_TOP) / BLOCK_H));
  const row1 = Math.min(OBS_ROWS - 1, Math.ceil((y + h - VIEW_TOP) / BLOCK_H) - 1);
  if (col1 < col0 || row1 < row0) return;
  const base = channel * OBS_ROWS * OBS_COLS;
  for (let row = row0; row <= row1; row++) {
    for (let col = col0; col <= col1; col++) {
      const index = base + row * OBS_COLS + col;
      grid[index] = Math.max(grid[index], value);
    }
  }
}

export function encodeObservation(sim, grid = new Uint8Array(OBS_SIZE), state = new Float32Array(STATE_SIZE)) {
  grid.fill(0);
  for (const block of sim.blocks.blocks) {
    const screenY = block.y + sim.camY;
    let channel;
    if (!block.fixed) channel = block.type === 'gravel' ? 4 : 3;
    else if (block.type === 'gravel') channel = 1;
    else if (block.type === 'beam') channel = 2;
    else channel = 0;
    const value = block.fixed ? 255 : Math.min(255, 64 + Math.round(block.yVel * 14));
    fillRect(grid, channel, block.x, screenY, block.w, block.h, value);
    if (block.faultTimer > 0) {
      const warning = Math.max(32, 255 - Math.round(255 * block.faultTimer / block.faultDuration));
      fillRect(grid, 5, block.x, screenY, block.w, block.h, warning);
    }
  }
  for (const forecast of sim.director.forecasts) {
    const material = forecast.type === 'gravel' ? 96 : forecast.type === 'beam' ? 176 : 255;
    fillRect(grid, 6, forecast.x, forecast.y + sim.camY, forecast.w, forecast.h, material);
  }

  const p = sim.player;
  fillRect(grid, 7, p.x, p.y + sim.camY, p.w, p.h, 255);
  state.set([
    (p.x + p.w / 2 - ARENA_X) / ARENA_W,
    (p.y + sim.camY - VIEW_TOP) / (OBS_ROWS * BLOCK_H),
    p.xVel / MOVE_SPEED,
    p.yVel / PLAYER_FALL_CAP,
    p.offGround === 0 ? 1 : 0,
    p.wallSide,
    p.lastWallJumpSide,
    p.focus / 3,
    p.focusProgress / 3,
    p.focusAimTimer / 90,
    p.focusTimer / 8,
    p.focusDX,
    p.focusDY,
    Math.min(1, (sim.blockRate ?? 0) / 7),
    sim.director.pressure ?? 0,
    Math.min(1, sim.height / (GAME_H * 10)),
  ]);
  return { grid, state };
}
