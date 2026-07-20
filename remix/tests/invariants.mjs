import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import * as constants from '../src/constants.js';
import { Sim, NEUTRAL_INPUT } from '../src/sim/sim.js';

const {
  ARENA_W,
  ARENA_X,
  BLOCK_H,
  FOCUS_AIM_MAX_FRAMES,
  FOCUS_RECHARGE_LAYERS,
  GROUND,
} = constants;
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
  p.xVel = 0;
  p.yVel = 0;
  p.offGround = 0;
  p.supportBlock = support;
  p.stableFrames = 0;
}

function strictOverlap(a, b) {
  return (
    a.x < b.x + b.w &&
    a.x + a.w > b.x &&
    a.y < b.y + b.h &&
    a.y + a.h > b.y
  );
}

function assertFixedTerrainIsDisjoint(sim, context = '') {
  const fixed = sim.blocks.blocks.filter((block) => block.fixed);
  for (let i = 0; i < fixed.length; i++) {
    const a = fixed[i];
    assert.ok(a.x >= ARENA_X, `${context} block ${a.id ?? a.idx} escaped left rail`);
    assert.ok(
      a.x + a.w <= ARENA_X + ARENA_W,
      `${context} block ${a.id ?? a.idx} escaped right rail`,
    );
    for (let j = i + 1; j < fixed.length; j++) {
      const b = fixed[j];
      assert.equal(
        strictOverlap(a, b),
        false,
        `${context} fixed overlap: ${JSON.stringify({
          a: { id: a.id, type: a.type, x: a.x, y: a.y, w: a.w, h: a.h },
          b: { id: b.id, type: b.type, x: b.x, y: b.y, w: b.w, h: b.h },
        })}`,
      );
    }
  }
}

function rounded(value) {
  return Number(value.toFixed(5));
}

function snapshot(sim, events) {
  const blocks = sim.blocks.blocks
    .map((block) => ({
      id: block.id ?? block.idx,
      type: block.type,
      x: rounded(block.x),
      y: rounded(block.y),
      xVel: rounded(block.xVel),
      yVel: rounded(block.yVel),
      fixed: block.fixed,
      faultTimer: rounded(block.faultTimer),
    }))
    .sort((a, b) => a.id - b.id);
  const forecasts = sim.director.forecasts.map((forecast) => ({
    type: forecast.type,
    x: forecast.x,
    y: rounded(forecast.y),
    frames: rounded(forecast.frames),
  }));
  return {
    hash: sim.hash(),
    blocks,
    forecasts,
    elapsedFrames: rounded(sim.director.elapsedFrames),
    spawnCount: sim.director.spawnCount,
    rejectedSpawns: sim.director.rejectedSpawns,
    events,
  };
}

function deterministicHistory(seed, frames = 3600) {
  const sim = new Sim(seed);
  sim.kill = () => {};
  const events = [];
  sim.events.tap = (name, payload) => {
    if (name === 'stormDrop') {
      events.push([name, sim.frame, payload.type, payload.x, payload.w]);
    } else if (name === 'branchShatter') {
      events.push([
        name,
        sim.frame,
        payload.root?.id ?? payload.root?.idx,
        payload.blocks.map((block) => block.id ?? block.idx),
      ]);
    }
  };

  for (let frame = 1; frame <= frames; frame++) {
    sim.step(N);
    if (frame % 30 === 0) assertFixedTerrainIsDisjoint(sim, `seed ${seed}, frame ${frame}`);
  }
  return snapshot(sim, events);
}

test('Focus Aim has an explicit 90-real-tick maximum', () => {
  assert.equal(FOCUS_AIM_MAX_FRAMES, 90);
  const sim = new Sim(0xf0c05, { director: false });
  sim.player.y = 120;
  const events = [];
  sim.events.tap = (name, payload) => events.push({ name, payload });

  for (let tick = 1; tick < FOCUS_AIM_MAX_FRAMES; tick++) {
    sim.step({
      ...N,
      focusPressed: tick === 1,
      focusHeld: true,
      focusDirX: 1,
    });
    assert.ok(sim.player.focusAimTimer > 0, `Aim committed early on held tick ${tick}`);
    assert.equal(sim.player.focusTimer, 0, `dash began early on held tick ${tick}`);
  }

  sim.step({ ...N, focusHeld: true, focusDirX: 1 });
  assert.equal(sim.player.focusAimTimer, 0, 'Aim did not commit on held tick 90');
  assert.ok(sim.player.focusTimer > 0, 'timeout did not start the committed dash');
  assert.equal(events.filter((event) => event.name === 'focusStart').length, 1);
  const end = events.find((event) => event.name === 'focusAimEnd');
  assert.equal(end?.payload.heldFrames, FOCUS_AIM_MAX_FRAMES);
});

test('a pending stable-height recharge cannot refund an active Aim', () => {
  const sim = new Sim(0xc0ffee, { director: false });
  const tower = [];
  for (let layer = 1; layer <= FOCUS_RECHARGE_LAYERS; layer++) {
    tower[layer] = fixedBlock(sim, 8, layer);
  }
  placeOnLayer(
    sim,
    8,
    FOCUS_RECHARGE_LAYERS,
    tower[FOCUS_RECHARGE_LAYERS],
  );
  sim.player.focus = 1;
  sim.player.nextFocusLayer = FOCUS_RECHARGE_LAYERS;
  sim.startFocusAim({ ...N, focusDirX: 1 });

  // Keep the footing explicitly stable so this isolates the recharge policy
  // from fractional slow-motion contact offsets.
  for (let tick = 1; tick <= 90; tick++) {
    sim.player.offGround = 0;
    sim.player.supportBlock = tower[FOCUS_RECHARGE_LAYERS];
    sim.updateFocusRecharge(0.1);
  }

  assert.ok(sim.player.focusAimTimer > 0, 'Aim ended before the recharge check');
  assert.equal(sim.player.focus, 0, 'active Aim refunded its spent charge');
  assert.equal(
    sim.player.nextFocusLayer,
    FOCUS_RECHARGE_LAYERS,
    'active Aim consumed the pending recharge milestone',
  );
});

test('overloaded branches shatter once and never become simulation projectiles', () => {
  const sim = new Sim(0x5a77e2, { director: false });
  fixedBlock(sim, 0, 1);
  const branch = [];
  for (let layer = 2; layer <= 7; layer++) branch.push(fixedBlock(sim, 3, layer));

  const shatters = [];
  let falls = 0;
  sim.events.tap = (name, payload) => {
    if (name === 'branchShatter') shatters.push(payload);
    if (name === 'blockFall') falls++;
  };
  assert.equal(sim.blocks.markFault(branch[0], 'overload'), true);

  for (let frame = 0; frame < 1000; frame++) sim.blocks.update();

  assert.equal(shatters.length, 1, `branch shattered ${shatters.length} times`);
  assert.equal(falls, 0, `branch emitted ${falls} gameplay blockFall events`);
  assert.equal(shatters[0].blocks.length, branch.length);
  assert.ok(branch.every((block) => !sim.blocks.blocks.includes(block)));
  assertFixedTerrainIsDisjoint(sim, 'after branch shatter');
});

test('equal-speed coincident drops cannot settle into duplicate terrain', () => {
  const sim = new Sim(0xe0a1, { director: false });
  const first = sim.blocks.spawnAt(220, 80, 'wood', { yVel: 0, shade: 0 });
  const second = sim.blocks.spawnAt(220, 80, 'wood', { yVel: 0, shade: 1 });

  for (let frame = 0; frame < 240 && (!first.fixed || !second.fixed); frame++) {
    sim.blocks.update();
    assertFixedTerrainIsDisjoint(sim, `coincident drops, frame ${frame}`);
  }

  assert.equal(first.fixed, true);
  assert.equal(second.fixed, true);
  assert.equal(strictOverlap(first, second), false);
  assert.notEqual(first.y, second.y, 'coincident drops occupied the same layer');
});

test('long seeded runs preserve disjoint terrain and reproduce complete state', () => {
  const first = deterministicHistory(0xdecafbad);
  const second = deterministicHistory(0xdecafbad);
  assert.deepEqual(second, first);
  assert.ok(first.spawnCount >= 100, `only ${first.spawnCount} committed drops`);
});

test('checkpoint snapshots restore complete deterministic simulation state', () => {
  const source = new Sim(0xc4ec7, { director: false });
  const base = fixedBlock(source, 8, 1, 'wood');
  fixedBlock(source, 8, 2, 'beam');
  source.blocks.markFault(base, 'carve');
  source.blocks.spawnAt(360, 40, 'gravel', { yVel: 3, shade: 1 });
  source.director.commitEntry({ col: 22, type: 'wood', w: 60 });
  source.player.x = base.x + 15;
  source.player.y = base.y - source.player.h;
  source.player.offGround = 0;
  source.player.supportBlock = base;
  source.player.focus = 1;
  source.player.focusProgress = 2;
  source.frame = 137;
  source.rng.next();

  const checkpoint = source.snapshot();
  const serialized = JSON.stringify(checkpoint);
  const first = new Sim(checkpoint.seed, { director: false }).restore(checkpoint);
  const second = new Sim(checkpoint.seed, { director: false }).restore(checkpoint);

  assert.deepEqual(first.snapshot(), checkpoint);
  assert.deepEqual(second.snapshot(), checkpoint);
  for (let frame = 0; frame < 90; frame++) {
    const input = frame % 24 < 12 ? { ...N, right: true } : { ...N, left: true };
    first.step(input);
    second.step(input);
  }
  assert.deepEqual(second.snapshot(), first.snapshot());
  assert.equal(JSON.stringify(checkpoint), serialized, 'restoring mutated the saved checkpoint');
});

test('active director steps stay within a practical frame-time budget', () => {
  const sim = new Sim(0x51eed);
  sim.kill = () => {};
  for (let frame = 0; frame < 240; frame++) sim.step(N);

  const samples = [];
  const totalStart = performance.now();
  for (let frame = 0; frame < 1800; frame++) {
    const start = performance.now();
    sim.step(N);
    samples.push(performance.now() - start);
  }
  const total = performance.now() - totalStart;
  samples.sort((a, b) => a - b);
  const p99 = samples[Math.floor(samples.length * 0.99)];

  assert.ok(total < 3000, `1,800 active steps took ${total.toFixed(1)}ms`);
  assert.ok(p99 < 25, `p99 simulation step took ${p99.toFixed(1)}ms`);
});

process.exitCode = failures ? 1 : 0;
