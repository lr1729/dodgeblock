import assert from 'node:assert/strict';
import {
  BLOCK_FALL_CAP,
  BLOCK_H,
  CARVE_WARNING_FRAMES,
  COLLAPSE_WARNING_FRAMES,
  FOCUS_AIM_MAX_FRAMES,
  FOCUS_AIM_WORLD_SCALE,
  FOCUS_CAP,
  FOCUS_FRAMES,
  FOCUS_RECHARGE_LAYERS,
  FOCUS_SPEED,
  GROUND,
  MOVE_SPEED,
  PLAYER_FALL_CAP,
  PLAYER_MAX_X,
  PLAYER_MIN_X,
} from '../src/constants.js';
import { Player } from '../src/sim/player.js';
import { Sim, NEUTRAL_INPUT } from '../src/sim/sim.js';
import { createInput } from '../src/input.js';

const N = NEUTRAL_INPUT;
let failures = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`ok ${name}`);
  } catch (error) {
    failures++;
    console.error(`FAIL ${name}`);
    console.error(error.stack ?? error);
  }
}

function approx(actual, expected, epsilon = 1e-6) {
  assert.ok(
    Math.abs(actual - expected) <= epsilon,
    `expected ${actual} to be within ${epsilon} of ${expected}`,
  );
}

function quietSim(seed = 1) {
  return new Sim(seed, { director: false });
}

function hardcoreSim(seed = 1) {
  return new Sim(seed, { director: false, rules: { autoGuard: false } });
}

function eventLog(sim) {
  const log = [];
  sim.events.tap = (name, payload) => log.push({ name, payload });
  return log;
}

function fixedBlock(sim, col, layer, type = 'wood') {
  const block = sim.blocks.spawnColumn(col, sim.camY, type, {
    y: GROUND.y - BLOCK_H * layer,
    yVel: 0,
    shade: 0,
  });
  const fallingIndex = sim.blocks.falling.indexOf(block);
  if (fallingIndex !== -1) sim.blocks.falling.splice(fallingIndex, 1);
  sim.blocks.fixAt(block, layer);
  return block;
}

function placeOnLayer(sim, col, layer, support) {
  const p = sim.player;
  p.x = sim.blocks.xForColumn(col) + 15;
  p.y = GROUND.y - BLOCK_H * layer - p.h;
  p.yVel = 0;
  p.offGround = 0;
  p.supportBlock = support;
  p.stableFrames = 0;
}

function touchInputHarness() {
  const pointerHandlers = new Map();
  const keyHandlers = new Map();
  const key = () => ({ isDown: false });
  const scene = {
    scale: { width: 800, height: 500 },
    sys: { game: { device: { input: { touch: true } } } },
    input: {
      keyboard: {
        addKeys: () => ({
          UP: key(), DOWN: key(), LEFT: key(), RIGHT: key(),
          W: key(), A: key(), S: key(), D: key(), SPACE: key(),
        }),
        on: (name, fn) => keyHandlers.set(name, fn),
        addCapture: () => {},
      },
      on: (name, fn) => pointerHandlers.set(name, fn),
    },
  };
  return { input: createInput(scene), pointerHandlers };
}

test('movement reaches speed quickly and reverses responsively', () => {
  const sim = quietSim();
  sim.step(N);
  const startX = sim.player.x;

  for (let i = 0; i < 4; i++) sim.step({ ...N, right: true });
  approx(sim.player.xVel, MOVE_SPEED);
  assert.ok(sim.player.x > startX + 10);

  for (let i = 0; i < 7; i++) sim.step({ ...N, left: true });
  approx(sim.player.xVel, -MOVE_SPEED);

  for (let i = 0; i < 3; i++) sim.step(N);
  approx(sim.player.xVel, 0);

  const airborne = new Player();
  airborne.offGround = 20;
  for (let i = 0; i < 5; i++) airborne.move(1);
  approx(airborne.xVel, MOVE_SPEED);
  airborne.xVel = 0;
  airborne.move(1, 0.1);
  approx(airborne.xVel, 0.12);
});

test('holding jump repeats whenever stable footing returns', () => {
  const sim = quietSim(101);
  let jumps = 0;
  sim.events.tap = (name) => {
    if (name === 'jump') jumps++;
  };
  sim.step(N);
  for (let i = 0; i < 150; i++) sim.step({ ...N, up: true });
  assert.ok(jumps >= 2, `held jump produced only ${jumps} jump(s)`);
});

test('touch jump zone produces one buffered jump press', () => {
  const { input, pointerHandlers } = touchInputHarness();
  pointerHandlers.get('pointerdown')({
    id: 1,
    x: 400,
    y: 80,
    isDown: true,
  });

  assert.equal(input.up, true);
  assert.equal(input.consumePressed().jumpPressed, true);
  assert.equal(input.consumePressed().jumpPressed, false);
});

test('touch swipe preserves an intentional zero direction axis', () => {
  const { input, pointerHandlers } = touchInputHarness();
  const pointer = { id: 2, x: 100, y: 400, isDown: true, downTime: performance.now() - 150 };
  pointerHandlers.get('pointerdown')(pointer);
  pointer.y = 345;
  pointerHandlers.get('pointermove')(pointer);

  const pressed = input.consumePressed();
  assert.equal(pressed.jumpPressed, false);
  assert.equal(pressed.focusPressed, true);
  assert.equal(pressed.focusDirX, 0);
  assert.equal(pressed.focusDirY, -1);
  const held = input.consumePressed();
  assert.equal(held.focusDirX, 0);
  assert.equal(held.focusDirY, -1);
  pointer.isDown = false;
  pointerHandlers.get('pointerup')(pointer);
  assert.equal(input.consumePressed().focusReleased, true);
});

test('a prompt vertical swipe starts Focus without the horizontal reversal delay', () => {
  const { input, pointerHandlers } = touchInputHarness();
  const pointer = { id: 4, x: 600, y: 400, isDown: true, downTime: performance.now() };
  pointerHandlers.get('pointerdown')(pointer);
  pointer.y = 330;
  pointerHandlers.get('pointermove')(pointer);
  const pressed = input.consumePressed();
  assert.equal(pressed.focusPressed, true);
  assert.equal(pressed.focusDirY, -1);
});

test('a rapid movement-thumb reversal does not trigger Focus', () => {
  const { input, pointerHandlers } = touchInputHarness();
  const pointer = { id: 3, x: 100, y: 400, isDown: true, downTime: performance.now() };
  pointerHandlers.get('pointerdown')(pointer);
  pointer.x = 500;
  pointerHandlers.get('pointermove')(pointer);
  assert.equal(input.consumePressed().focusPressed, false);
});

test('Focus travels in all four directions and normalizes diagonals', () => {
  const cases = [
    { dx: 1, dy: 0, sx: 1, sy: 0 },
    { dx: -1, dy: 0, sx: -1, sy: 0 },
    { dx: 0, dy: -1, sx: 0, sy: -1 },
    { dx: 0, dy: 1, sx: 0, sy: 1 },
    { dx: 1, dy: -1, sx: 1, sy: -1 },
  ];

  for (const direction of cases) {
    const sim = quietSim(2);
    sim.player.y = 120;
    const x = sim.player.x;
    const y = sim.player.y;
    sim.step({
      ...N,
      focusPressed: true,
      focusDirX: direction.dx,
      focusDirY: direction.dy,
    });
    const movedX = sim.player.x - x;
    const movedY = sim.player.y - y;
    assert.equal(Math.sign(movedX), direction.sx);
    assert.equal(Math.sign(movedY), direction.sy);
    approx(Math.hypot(movedX, movedY), FOCUS_SPEED, 1e-5);
    assert.equal(sim.player.focus, FOCUS_CAP - 1);
  }
});

test('cardinal gesture direction is not polluted by held controls', () => {
  const sim = quietSim(21);
  sim.player.y = 120;
  sim.step({
    ...N,
    up: true,
    focusPressed: true,
    focusDirX: 1,
    focusDirY: 0,
  });
  assert.equal(sim.player.focusDX, 1);
  assert.equal(sim.player.focusDY, 0);
});

test('Focus Aim preserves the last nonzero direction after its key is released', () => {
  const sim = quietSim(211);
  sim.player.y = 120;
  sim.step({ ...N, up: true, focusPressed: true, focusHeld: true });
  assert.equal(sim.player.focusDX, 0);
  assert.equal(sim.player.focusDY, -1);

  sim.step({ ...N, focusHeld: true });
  assert.equal(sim.player.focusDX, 0);
  assert.equal(sim.player.focusDY, -1);

  const y = sim.player.y;
  sim.step({ ...N, focusReleased: true });
  assert.ok(sim.player.y < y, 'releasing Aim reverted to horizontal facing');
});

test('horizontal Focus works while grounded or standing on a block', () => {
  const groundSim = quietSim(22);
  groundSim.step(N);
  const groundX = groundSim.player.x;
  groundSim.step({ ...N, focusPressed: true, focusDirX: 1 });
  assert.ok(groundSim.player.x > groundX);
  assert.ok(groundSim.player.focusTimer > 0);

  const blockSim = quietSim(23);
  const support = fixedBlock(blockSim, 5, 1, 'wood');
  placeOnLayer(blockSim, 5, 1, support);
  const blockX = blockSim.player.x;
  blockSim.step({ ...N, focusPressed: true, focusDirX: 1 });
  assert.ok(blockSim.player.x > blockX);
  assert.ok(blockSim.blocks.blocks.includes(support));
});

test('held Focus Aim keeps player and world motion in slow time, then forces a dash', () => {
  const normal = quietSim(3);
  const normalBlock = normal.blocks.spawnAt(50, -800, 'wood', { yVel: 1, shade: 0 });
  const normalY = normalBlock.y;
  normal.step(N);

  const focused = quietSim(3);
  focused.player.y = 120;
  const focusedBlock = focused.blocks.spawnAt(50, -800, 'wood', { yVel: 1, shade: 0 });
  const focusedY = focusedBlock.y;
  const playerX = focused.player.x;
  const playerY = focused.player.y;
  focused.player.xVel = 4;
  focused.player.yVel = -3;
  focused.step({
    ...N,
    right: true,
    focusPressed: true,
    focusHeld: true,
    focusDirX: 1,
  });

  const normalFall = normalBlock.y - normalY;
  const focusedFall = focusedBlock.y - focusedY;
  assert.ok(focusedFall < normalFall * 0.25, `${focusedFall} vs ${normalFall}`);
  assert.ok(focusedFall > normalFall * FOCUS_AIM_WORLD_SCALE * 0.7);
  assert.ok(focused.player.x > playerX, 'Aim pinned horizontal player motion');
  assert.ok(focused.player.x - playerX < 1, 'Aim did not slow horizontal player motion');
  assert.ok(focused.player.y > playerY, 'Aim pinned vertical player motion');
  assert.ok(focused.player.y - playerY < 1, 'Aim did not slow vertical player motion');
  assert.ok(focused.player.focusAimTimer > 0);

  const beforeDashX = focused.player.x;
  focused.step({ ...N, focusReleased: true, focusDirX: 1 });
  approx(focused.player.x - beforeDashX, FOCUS_SPEED);
  assert.equal(focused.player.focusAimTimer, 0);
  assert.ok(focused.player.focusTimer > 0);
});

test('Focus shatters exactly one falling block and stops', () => {
  const sim = quietSim(4);
  const log = eventLog(sim);
  const first = sim.blocks.spawnAt(260, 100, 'wood', { yVel: 0, shade: 0 });
  const second = sim.blocks.spawnAt(260, 100, 'wood', { yVel: 0, shade: 1 });
  const p = sim.player;
  p.x = 225;
  p.y = 105;
  p.offGround = 10;

  sim.step({ ...N, focusPressed: true, focusDirX: 1 });

  assert.equal(sim.blocks.blocks.includes(first), false, 'hit block survived');
  assert.equal(sim.blocks.blocks.includes(second), true, 'Focus affected a second block');
  assert.equal(sim.player.focusTimer, 0);
  assert.equal(log.filter((entry) => entry.name === 'focusKick').length, 1);
});

test('Focus selects the earliest geometric hit instead of the first array entry', () => {
  const sim = quietSim(41);
  const far = sim.blocks.spawnAt(142, 100, 'wood', { yVel: 0, shade: 0 });
  const near = sim.blocks.spawnAt(138, 100, 'wood', { yVel: 0, shade: 1 });
  const p = sim.player;
  p.x = 100;
  p.y = 100;
  p.focusDX = 1;
  p.focusDY = 0;
  p.focusTimer = FOCUS_FRAMES;

  sim.updateFocus();
  sim.updateFocus();

  assert.equal(sim.blocks.blocks.includes(near), false, 'nearer block survived');
  assert.equal(sim.blocks.blocks.includes(far), true, 'farther array-first block was selected');
});

test('Focus does not hit a touching block when moving away from it', () => {
  const sim = quietSim(42);
  const block = sim.blocks.spawnAt(100, 100, 'wood', { yVel: 0, shade: 0 });
  const p = sim.player;
  p.x = block.x + block.w;
  p.y = block.y;
  p.focusDX = 1;
  p.focusDY = 0;
  p.focusTimer = FOCUS_FRAMES;

  sim.updateFocus();

  assert.equal(sim.blocks.blocks.includes(block), true);
  assert.ok(p.x > block.x + block.w, 'Focus failed to move away from the touching block');
});

test('Focus cuts settled terrain from all eight directions', () => {
  const approaches = [
    { dx: 1, dy: 0, place: (p, b) => [b.x - p.w - 4, b.y + 5] },
    { dx: -1, dy: 0, place: (p, b) => [b.x + b.w + 4, b.y + 5] },
    { dx: 0, dy: 1, place: (p, b) => [b.x + 15, b.y - p.h - 4] },
    { dx: 0, dy: -1, place: (p, b) => [b.x + 15, b.y + b.h + 4] },
    { dx: 1, dy: 1, place: (p, b) => [b.x - p.w - 4, b.y - p.h - 4] },
    { dx: -1, dy: 1, place: (p, b) => [b.x + b.w + 4, b.y - p.h - 4] },
    { dx: 1, dy: -1, place: (p, b) => [b.x - p.w - 4, b.y + b.h + 4] },
    { dx: -1, dy: -1, place: (p, b) => [b.x + b.w + 4, b.y + b.h + 4] },
  ];

  for (const [index, approach] of approaches.entries()) {
    const sim = quietSim(401 + index);
    const log = eventLog(sim);
    const wood = fixedBlock(sim, 20, 5, 'wood');
    const p = sim.player;
    [p.x, p.y] = approach.place(p, wood);
    const magnitude = Math.hypot(approach.dx, approach.dy);
    p.focusDX = approach.dx / magnitude;
    p.focusDY = approach.dy / magnitude;
    p.focusTimer = FOCUS_FRAMES;

    sim.updateFocus();

    assert.ok(wood.faultTimer > 0, `direction ${approach.dx},${approach.dy} did not cut`);
    assert.ok(log.some(
      (entry) => entry.name === 'focusEnd' && entry.payload.reason === 'break',
    ));
  }
});

test('an upward Focus marks the supported overhead branch, then shatters it atomically', () => {
  const sim = quietSim(402);
  // A left shoulder supports the overhang while leaving space underneath it.
  fixedBlock(sim, 17, 1, 'wood');
  const lower = fixedBlock(sim, 20, 2, 'wood');
  const upper = fixedBlock(sim, 20, 3, 'wood');
  const p = sim.player;
  p.x = lower.x + 15;
  p.y = lower.y + lower.h + 4;
  p.offGround = 20;

  sim.step({ ...N, focusPressed: true, focusDirY: -1 });

  assert.equal(sim.blocks.blocks.includes(lower), true);
  assert.equal(sim.blocks.blocks.includes(upper), true);
  assert.ok(lower.faultTimer > 0);
  assert.ok(upper.faultTimer > 0);
  assert.equal(p.focusTimer, 0);
  for (let i = 0; i < COLLAPSE_WARNING_FRAMES; i++) sim.blocks.update();
  assert.equal(sim.blocks.blocks.includes(lower), false);
  assert.equal(sim.blocks.blocks.includes(upper), false);
});

test('Focus Aim spends its charge immediately and auto-commits after 1.5 seconds', () => {
  const sim = quietSim(5);
  const startX = sim.player.x;
  sim.step({ ...N, focusPressed: true, focusHeld: true, focusDirX: 1 });
  assert.equal(sim.player.focus, FOCUS_CAP - 1);
  for (let i = 1; i < FOCUS_AIM_MAX_FRAMES - 1; i++) {
    sim.step({ ...N, focusHeld: true, focusDirX: 1 });
  }
  assert.ok(sim.player.focusAimTimer > 0);
  approx(sim.player.x, startX);
  sim.step({ ...N, focusHeld: true, focusDirX: 1 });
  assert.equal(sim.player.focusAimTimer, 0);
  assert.ok(sim.player.x > startX, 'timeout did not commit the dash');
});

test('a falling block kills only on a descending overhead contact', () => {
  const sim = hardcoreSim(24);
  sim.player.x = 200;
  sim.player.y = 100;
  sim.player.offGround = 10;
  sim.blocks.spawnAt(190, 65, 'wood', { yVel: 4, shade: 0 });
  sim.step(N);
  assert.equal(sim.dead, true);
});

test('a block that seats on terrain still crushes the player on its impact frame', () => {
  const sim = hardcoreSim(243);
  const p = sim.player;
  p.x = 200;
  p.y = GROUND.y - p.h;
  p.offGround = 0;
  const wood = sim.blocks.spawnAt(190, 259, 'wood', { yVel: 4, shade: 0 });

  sim.step(N);

  assert.equal(wood.fixed, true);
  assert.equal(sim.dead, true);
  assert.equal(sim.deathCause, 'squished');
});

test('side contact with falling wood pushes instead of killing', () => {
  const sim = quietSim(241);
  const p = sim.player;
  p.x = 200;
  p.y = 100;
  p.offGround = 10;
  sim.blocks.spawnAt(225, 100, 'wood', { yVel: 4, shade: 0 });

  sim.step(N);

  assert.equal(sim.dead, false);
  assert.ok(p.x < 200, `side contact did not push the player: ${p.x}`);
});

test('top contact with falling wood remains physically supported', () => {
  const sim = quietSim(242);
  const p = sim.player;
  p.x = 200;
  p.y = 100;
  p.yVel = -2;
  p.offGround = 10;
  const wood = sim.blocks.spawnAt(190, 125, 'wood', { yVel: 1, shade: 0 });

  sim.step(N);

  assert.equal(sim.dead, false);
  assert.equal(p.supportBlock, wood);
  assert.equal(wood.fixed, false);
});

test('Auto Guard spends one Focus charge for a simultaneous crush incident', () => {
  const sim = quietSim(244);
  const log = eventLog(sim);
  const p = sim.player;
  p.x = 200;
  p.y = 100;
  p.offGround = 10;
  const first = sim.blocks.spawnAt(190, 65, 'wood', { yVel: 4, shade: 0 });
  const second = sim.blocks.spawnAt(205, 65, 'wood', { yVel: 4, shade: 1 });

  sim.frame = 1;
  first.previousY = 65;
  first.y = 70;
  second.previousY = 65;
  second.y = 70;
  sim.resolveIncomingBlockMotion();

  assert.equal(sim.dead, false);
  assert.equal(p.focus, FOCUS_CAP - 1);
  assert.equal(sim.blocks.blocks.includes(first), false);
  assert.equal(sim.blocks.blocks.includes(second), false);
  assert.equal(log.filter(({ name }) => name === 'autoGuard').length, 1);
});

test('an Aim charge converts into Auto Guard without charging twice', () => {
  const sim = quietSim(245);
  const p = sim.player;
  p.x = 200;
  p.y = 100;
  p.offGround = 10;
  sim.step({ ...N, focusPressed: true, focusHeld: true, focusDirX: 1 });
  assert.equal(p.focus, FOCUS_CAP - 1);
  sim.blocks.spawnAt(190, 65, 'wood', { yVel: 4, shade: 0 });

  sim.step({ ...N, focusHeld: true, focusDirX: 1 });

  assert.equal(sim.dead, false);
  assert.equal(p.focus, FOCUS_CAP - 1);
  assert.equal(p.focusAimTimer, 0);
});

test('a just-committed Aim charge still guards its release frame', () => {
  const sim = quietSim(246);
  const p = sim.player;
  p.focus = 1;
  p.x = 200;
  p.y = 100;
  p.offGround = 10;
  sim.step({ ...N, focusPressed: true, focusHeld: true, focusDirX: 1 });
  assert.equal(p.focus, 0);
  sim.blocks.spawnAt(190, 65, 'wood', { yVel: 4, shade: 0 });

  sim.step({ ...N, focusReleased: true, focusDirX: 1 });

  assert.equal(sim.dead, false);
  assert.equal(p.focus, 0);
  assert.equal(p.focusTimer, 0, 'guard did not cancel the pending dash');
});

test('a later crush consumes another charge despite visual recovery', () => {
  const sim = quietSim(247);
  const p = sim.player;
  p.x = 200;
  p.y = 100;
  p.offGround = 10;
  sim.blocks.spawnAt(190, 65, 'wood', { yVel: 4, shade: 0 });
  sim.step(N);
  assert.equal(p.focus, FOCUS_CAP - 1);

  p.x = 200;
  p.y = 100;
  p.offGround = 10;
  sim.blocks.spawnAt(190, 65, 'wood', { yVel: 4, shade: 0 });
  sim.step(N);

  assert.equal(sim.dead, false);
  assert.equal(p.focus, 0);
});

test('Focus bonks on the ground', () => {
  const sim = quietSim(6);
  const log = eventLog(sim);
  const p = sim.player;
  p.y = GROUND.y - p.h;
  p.offGround = 0;
  const y = p.y;

  sim.step({ ...N, focusPressed: true, focusDirY: 1 });

  approx(p.y, y);
  assert.equal(p.focusTimer, 0);
  assert.ok(log.some(
    (entry) => entry.name === 'focusEnd' && entry.payload.reason === 'bonk',
  ));
});

test('Focus bonks at the arena edge', () => {
  const sim = quietSim(61);
  const log = eventLog(sim);
  const p = sim.player;
  p.x = PLAYER_MAX_X - p.w;
  p.y = 120;

  sim.step({ ...N, focusPressed: true, focusDirX: 1 });

  assert.equal(p.x, PLAYER_MAX_X - p.w);
  assert.equal(p.focusTimer, 0);
  assert.ok(log.some(
    (entry) => entry.name === 'focusEnd' && entry.payload.reason === 'bonk',
  ));
});

test('wall jump launches away once until another wall or landing resets it', () => {
  const sim = quietSim(62);
  const p = sim.player;
  p.y = 120;
  p.offGround = 20;
  p.wallSide = 1;

  sim.step({ ...N, up: true, jumpPressed: true });
  assert.ok(p.xVel < 0, `wall jump did not launch left: ${p.xVel}`);
  assert.equal(p.lastWallJumpSide, 1);

  p.offGround = 20;
  p.wallSide = 1;
  p.timeSinceJump = 10;
  const yVel = p.yVel;
  assert.equal(p.jump(), false);
  assert.equal(p.yVel, yVel);

  p.wallSide = -1;
  assert.equal(p.jump(), true);
  assert.ok(p.xVel > 0);
});

test('wall contact side comes from geometry even without horizontal motion', () => {
  const sim = quietSim(621);
  const block = fixedBlock(sim, 10, 3, 'wood');
  const p = sim.player;
  p.y = block.y + 5;
  p.xVel = 0;

  p.x = block.x - p.w;
  p.wallSide = 0;
  sim.wallPass();
  assert.equal(p.wallSide, 1, 'block on the right produced the wrong wall side');

  p.x = block.x + block.w;
  p.wallSide = 0;
  sim.wallPass();
  assert.equal(p.wallSide, -1, 'block on the left produced the wrong wall side');
});

test('horizontal collisions project to exact faces independent of fractional approach', () => {
  for (const start of [101.25, 103.75, 106.5]) {
    const sim = quietSim(620);
    const block = fixedBlock(sim, 10, 3, 'wood');
    const p = sim.player;
    p.y = block.y + 5;
    p.x = block.x - p.w - start % 4 - 1;
    const fromX = p.x;
    p.x += 8;
    sim.resolvePlayerX(fromX);
    approx(p.x, block.x - p.w);
  }
});

test('horizontal collision result does not depend on block array order', () => {
  const run = (reverse) => {
    const sim = quietSim(623);
    const first = fixedBlock(sim, 10, 3, 'wood');
    fixedBlock(sim, 14, 3, 'wood');
    if (reverse) sim.blocks.blocks.reverse();
    const p = sim.player;
    p.y = first.y + 5;
    p.x = first.x - p.w - 5;
    const fromX = p.x;
    p.x += 12;
    sim.resolvePlayerX(fromX);
    return { x: p.x, xVel: p.xVel, wallSide: p.wallSide };
  };
  assert.deepEqual(run(true), run(false));
});

test('wall jump remains available for four grace frames after leaving contact', () => {
  const p = new Player();
  p.offGround = 20;
  p.timeSinceJump = 10;
  p.rememberWall(1);
  p.wallSide = 0;
  for (let i = 0; i < 3; i++) p.updateWallCoyote();
  assert.equal(p.jump(), true);
  assert.ok(p.xVel < 0);

  const expired = new Player();
  expired.offGround = 20;
  expired.timeSinceJump = 10;
  expired.rememberWall(1);
  expired.wallSide = 0;
  for (let i = 0; i < 4; i++) expired.updateWallCoyote();
  assert.equal(expired.jump(), false);
});

test('a small rising head-corner overlap is corrected without cancelling the jump', () => {
  const sim = quietSim(622);
  const ceiling = fixedBlock(sim, 10, 3, 'wood');
  const p = sim.player;
  p.x = ceiling.x - p.w + 3;
  p.y = ceiling.y + ceiling.h + 1;
  p.yVel = 6;
  p.offGround = 20;
  const startX = p.x;

  sim.step({ ...N, up: true });

  assert.ok(p.x < startX, `corner correction did not move left: ${startX} -> ${p.x}`);
  assert.ok(p.yVel > 0, `corner correction cancelled upward velocity: ${p.yVel}`);
});

test('a small falling ledge overlap follows horizontal intent into an open gap', () => {
  const sim = quietSim(624);
  const ledge = fixedBlock(sim, 10, 3, 'wood');
  const p = sim.player;
  const fromY = ledge.y - p.h - 8;
  p.x = ledge.x - p.w + 3;
  p.y = fromY + 12;
  p.xVel = -3;
  p.yVel = -6;
  p.offGround = 20;

  sim.resolvePlayerY(fromY);

  approx(p.x, ledge.x - p.w);
  approx(p.y, fromY + 12);
  assert.ok(p.offGround > 0, 'gap assist incorrectly landed on the ledge');
});

test('falling ledge correction requires matching horizontal intent', () => {
  const sim = quietSim(625);
  const ledge = fixedBlock(sim, 10, 3, 'wood');
  const p = sim.player;
  const fromY = ledge.y - p.h - 8;
  p.x = ledge.x - p.w + 3;
  p.y = fromY + 12;
  p.xVel = 0;
  p.yVel = -6;

  sim.resolvePlayerY(fromY);

  approx(p.y, ledge.y - p.h);
  assert.equal(p.offGround, 0);
});

test('breaking gravel warns its whole dependent branch before atomic shatter', () => {
  const sim = quietSim(7);
  const gravel = fixedBlock(sim, 5, 1, 'gravel');
  const upper = fixedBlock(sim, 5, 2, 'wood');
  const p = sim.player;
  p.x = sim.blocks.xForColumn(5) - p.w + 4;
  p.y = GROUND.y - p.h;
  p.offGround = 0;

  sim.step({ ...N, focusPressed: true, focusDirX: 1 });
  assert.equal(gravel.fixed, true);
  assert.equal(upper.fixed, true);
  assert.equal(gravel.faultRoot, gravel);
  assert.equal(upper.faultRoot, gravel);
  assert.equal(gravel.faultTimer, CARVE_WARNING_FRAMES);
  for (let i = 0; i < COLLAPSE_WARNING_FRAMES; i++) sim.blocks.update();
  assert.equal(sim.blocks.blocks.includes(gravel), false);
  assert.equal(sim.blocks.blocks.includes(upper), false);
});

test('overlapping branch warnings merge into one earliest atomic shatter', () => {
  const sim = quietSim(71);
  const log = eventLog(sim);
  const low = fixedBlock(sim, 4, 1, 'wood');
  const mid = fixedBlock(sim, 4, 2, 'wood');
  const top = fixedBlock(sim, 4, 3, 'wood');
  sim.blocks.markFault(mid, 'overload');
  for (let i = 0; i < 8; i++) sim.blocks.update();
  const earlierTimer = mid.faultTimer;
  sim.blocks.markFault(low, 'carve');
  assert.equal(sim.blocks.faults.length, 1);
  assert.equal(sim.blocks.faults[0], low);
  assert.equal(low.faultTimer, Math.min(earlierTimer, CARVE_WARNING_FRAMES));
  while (sim.blocks.faults.length) sim.blocks.update();
  assert.equal(log.filter((entry) => entry.name === 'branchShatter').length, 1);
  assert.ok([low, mid, top].every((block) => !sim.blocks.blocks.includes(block)));
});

test('a warned fault remains solid support until it actually collapses', () => {
  const sim = quietSim(25);
  const fault = fixedBlock(sim, 2, 1, 'wood');
  sim.blocks.markFault(fault, 'overload');
  const newcomer = sim.blocks.spawnColumn(2, sim.camY, 'wood', {
    y: fault.y - BLOCK_H + 1,
    yVel: 0,
    shade: 0,
  });

  sim.blocks.update();
  assert.ok(fault.faultTimer > 0);
  assert.equal(newcomer.fixed, true, 'a block fell through visible warned terrain');
  assert.equal(newcomer.y, fault.y - BLOCK_H);
});

test('Focus slow motion also slows structural warning timers', () => {
  const sim = quietSim(250);
  const fault = fixedBlock(sim, 2, 1, 'wood');
  sim.blocks.markFault(fault, 'overload');
  const before = fault.faultTimer;

  sim.step({ ...N, focusPressed: true, focusHeld: true, focusDirX: 1 });

  assert.ok(fault.faultTimer < before);
  assert.ok(
    before - fault.faultTimer < 0.2,
    `collapse warning advanced ${before - fault.faultTimer} frames during Focus`,
  );
});

test('a tall tower on a narrow shoulder remains deterministic until deliberately carved', () => {
  const sim = quietSim(251);
  fixedBlock(sim, 0, 1, 'wood');
  const shoulder = fixedBlock(sim, 3, 2, 'wood');
  for (let layer = 3; layer <= 6; layer++) fixedBlock(sim, 3, layer, 'wood');

  sim.blocks.update();

  assert.equal(shoulder.faultTimer, 0);
  assert.equal(shoulder.fixed, true);
  assert.equal(sim.blocks.faults.length, 0);
});

test('a marked branch shatters once without becoming falling projectiles', () => {
  const sim = quietSim(254);
  const log = eventLog(sim);
  fixedBlock(sim, 0, 1, 'wood');
  const branch = [];
  for (let layer = 2; layer <= 6; layer++) branch.push(fixedBlock(sim, 3, layer, 'wood'));
  sim.blocks.markFault(branch[0], 'carve');
  for (let i = 0; i < COLLAPSE_WARNING_FRAMES + 60; i++) sim.blocks.update();
  assert.equal(log.filter((entry) => entry.name === 'blockFault').length, 1);
  assert.equal(log.filter((entry) => entry.name === 'branchShatter').length, 1);
  assert.equal(log.filter((entry) => entry.name === 'blockFall').length, 0);
  assert.ok(branch.every((b) => !sim.blocks.blocks.includes(b)));
});

test('new terrain supported by a warning branch joins the same atomic shatter', () => {
  const sim = quietSim(255);
  const root = fixedBlock(sim, 8, 1, 'gravel');
  const upper = fixedBlock(sim, 8, 2, 'wood');
  sim.blocks.markFault(root, 'carve');
  const newcomer = sim.blocks.spawnColumn(8, sim.camY, 'wood', {
    y: upper.y - BLOCK_H + 1,
    yVel: 0,
    shade: 0,
  });
  sim.blocks.update();
  assert.equal(newcomer.fixed, true);
  sim.blocks.update();
  assert.equal(newcomer.faultRoot, root);
  while (root.faultTimer > 0) sim.blocks.update();
  assert.equal(sim.blocks.blocks.includes(newcomer), false);
});

test('alternate support excludes a bridge from a carved branch', () => {
  const sim = quietSim(252);
  const left = fixedBlock(sim, 0, 1, 'gravel');
  fixedBlock(sim, 4, 1, 'wood');
  const bridge = fixedBlock(sim, 2, 2, 'beam');
  sim.blocks.markFault(left, 'carve');
  assert.equal(left.faultRoot, left);
  assert.equal(bridge.faultRoot, null);
  for (let i = 0; i < COLLAPSE_WARNING_FRAMES; i++) sim.blocks.update();
  assert.equal(sim.blocks.blocks.includes(left), false);
  assert.equal(sim.blocks.blocks.includes(bridge), true);
});

test('structural cuts occur only after an explicit player action', () => {
  const sim = quietSim(253);
  const gravel = fixedBlock(sim, 0, 1, 'gravel');
  const aboveSeam = fixedBlock(sim, 0, 2, 'wood');
  assert.equal(gravel.faultTimer, 0);
  sim.blocks.markFault(gravel, 'carve');
  assert.equal(aboveSeam.faultRoot, gravel);
});

test('Focus recharges immediately on valid new footing every three layers', () => {
  const sim = quietSim(8);
  const tower = [];
  for (let layer = 1; layer <= 2; layer++) {
    tower[layer] = fixedBlock(sim, 6, layer, 'wood');
  }
  const p = sim.player;
  p.focus = 0;
  p.nextFocusLayer = FOCUS_RECHARGE_LAYERS;

  placeOnLayer(sim, 6, 2, tower[2]);
  sim.step(N);
  assert.equal(p.focus, 0, 'recharged below the three-layer threshold');

  tower[3] = fixedBlock(sim, 6, 3, 'wood');
  placeOnLayer(sim, 6, 3, tower[3]);
  sim.step(N);
  assert.equal(p.focus, 1);
  assert.equal(p.nextFocusLayer, 6);

  tower[4] = fixedBlock(sim, 6, 4, 'wood');
  tower[5] = fixedBlock(sim, 6, 5, 'wood');
  placeOnLayer(sim, 6, 5, tower[5]);
  sim.step(N);
  assert.equal(p.focus, 1, 'recharged before gaining three more layers');

  tower[6] = fixedBlock(sim, 6, 6, 'wood');
  placeOnLayer(sim, 6, 6, tower[6]);
  sim.step(N);
  assert.equal(p.focus, 2);
  assert.equal(p.nextFocusLayer, 9);
});

test('held auto-hop earns recharge on the landing frame', () => {
  const sim = quietSim(801);
  fixedBlock(sim, 6, 1, 'wood');
  fixedBlock(sim, 6, 2, 'wood');
  const top = fixedBlock(sim, 6, 3, 'wood');
  const p = sim.player;
  p.focus = 0;
  p.x = top.x + 15;
  p.y = top.y - p.h - 1;
  p.yVel = -3;
  p.offGround = 20;

  sim.step({ ...N, up: true });
  assert.equal(p.focus, 1, 'valid landing did not recharge before auto-hop');
  assert.equal(p.offGround, 0);

  sim.step({ ...N, up: true });
  assert.ok(p.offGround > 0, 'held jump did not continue after the recharge landing');
});

test('full Focus discards recharge progress instead of banking a reserve', () => {
  const sim = quietSim(83);
  const tower = [];
  for (let layer = 1; layer <= 15; layer++) tower[layer] = fixedBlock(sim, 9, layer);
  placeOnLayer(sim, 9, 12, tower[12]);
  sim.step(N);
  assert.equal(sim.player.focus, FOCUS_CAP);
  assert.equal(sim.player.focusProgress, 0);

  sim.player.focus--;
  sim.step(N);
  assert.equal(sim.player.focus, FOCUS_CAP - 1);
  assert.equal(sim.player.focusProgress, 0);

  placeOnLayer(sim, 9, 14, tower[14]);
  sim.step(N);
  assert.equal(sim.player.focus, FOCUS_CAP - 1);
  assert.equal(sim.player.focusProgress, 2);

  placeOnLayer(sim, 9, 15, tower[15]);
  sim.step(N);
  assert.equal(sim.player.focus, FOCUS_CAP);
  assert.equal(sim.player.focusProgress, 0);
});

test('jump height does not increase score or stable-layer recharge progress', () => {
  const sim = quietSim(81);
  sim.step(N);
  const height = sim.height;
  const progress = sim.player.focusProgress;
  for (let i = 0; i < 40; i++) sim.step({ ...N, up: true });
  assert.equal(sim.height, height);
  assert.equal(sim.player.focusProgress, progress);
});

test('fault-warning footing cannot recharge Focus', () => {
  const sim = quietSim(26);
  fixedBlock(sim, 7, 1, 'wood');
  fixedBlock(sim, 7, 2, 'wood');
  const top = fixedBlock(sim, 7, 3, 'wood');
  sim.player.focus = 0;
  sim.player.nextFocusLayer = 3;
  sim.blocks.markFault(top, 'overload');
  placeOnLayer(sim, 7, 3, top);

  sim.step(N);
  assert.equal(sim.player.focus, 0);
  assert.equal(sim.player.stableFrames, 0);
});

test('player and block velocities respect terminal caps and arena bounds', () => {
  const player = new Player();
  player.offGround = 100;
  player.y = -1000;
  player.yVel = -100;
  player.updateY(false, true);
  assert.equal(player.yVel, -PLAYER_FALL_CAP);

  for (let i = 0; i < 20; i++) player.move(1);
  assert.equal(player.xVel, MOVE_SPEED);
  player.x = PLAYER_MAX_X;
  player.updateX();
  assert.equal(player.x, PLAYER_MAX_X - player.w);
  assert.equal(player.xVel, 0);
  player.xVel = -MOVE_SPEED;
  player.x = PLAYER_MIN_X - 50;
  player.updateX();
  assert.equal(player.x, PLAYER_MIN_X);

  const sim = quietSim(11);
  const wood = sim.blocks.spawnAt(50, -3000, 'wood', { yVel: 0, shade: 0 });
  const gravel = sim.blocks.spawnAt(170, -3000, 'gravel', { yVel: 0, shade: 0 });
  for (let i = 0; i < 240; i++) sim.blocks.update();
  approx(wood.yVel, BLOCK_FALL_CAP);
  approx(gravel.yVel, BLOCK_FALL_CAP);
});

test('falling-block pushout at a rail chooses the open in-bounds side', () => {
  const sim = quietSim(111);
  const p = sim.player;
  p.x = PLAYER_MAX_X - p.w;
  p.y = 100;
  p.xVel = MOVE_SPEED;
  p.offGround = 20;
  sim.blocks.spawnAt(730, 100, 'gravel', { yVel: 1, shade: 0 });

  sim.step({ ...N, right: true });

  assert.equal(p.x, 700);
  assert.ok(p.x >= PLAYER_MIN_X && p.x + p.w <= PLAYER_MAX_X);
  assert.equal(p.xVel, 0);
});

test('ceiling contact cancels vertical motion without killing air control', () => {
  const sim = quietSim(27);
  const ceiling = fixedBlock(sim, 5, 3, 'wood');
  const p = sim.player;
  p.x = ceiling.x + 10;
  p.y = ceiling.y + ceiling.h + 1;
  p.yVel = 6;
  p.offGround = 20;

  for (let i = 0; i < 3; i++) sim.step({ ...N, right: true, up: true });
  assert.ok(p.y >= ceiling.y + ceiling.h, `player remained inside ceiling at ${p.y}`);
  assert.ok(p.xVel >= 3.5, `ceiling contact erased horizontal control: ${p.xVel}`);
  assert.ok(p.yVel <= 0, `upward velocity survived ceiling contact: ${p.yVel}`);
});

test('old terrain remains solid after more than 360 newer blocks', () => {
  const sim = quietSim(12);
  const oldSupport = fixedBlock(sim, 4, 1, 'wood');
  for (let i = 0; i < 361; i++) fixedBlock(sim, 11, i + 1, 'wood');

  const p = sim.player;
  p.x = sim.blocks.xForColumn(4) + 15;
  p.y = oldSupport.y - p.h - 1;
  p.yVel = -3;
  p.offGround = 20;
  sim.step(N);

  approx(p.y, oldSupport.y - p.h);
  assert.equal(p.supportBlock, oldSupport);
});

test('old terrain participates in the same warned atomic carve transaction', () => {
  const sim = quietSim(13);
  const oldSupport = fixedBlock(sim, 0, 1, 'wood');
  const oldUpper = fixedBlock(sim, 0, 2, 'wood');
  for (let i = 0; i < 321; i++) fixedBlock(sim, 11, i + 1, 'wood');

  sim.blocks.removeFixedAndCollapse(oldSupport);
  assert.equal(oldUpper.fixed, true);
  assert.equal(oldUpper.faultRoot, oldSupport);
  for (let i = 0; i < COLLAPSE_WARNING_FRAMES; i++) sim.blocks.update();
  assert.equal(sim.blocks.blocks.includes(oldUpper), false);
});

process.exitCode = failures ? 1 : 0;
