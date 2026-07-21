import assert from 'node:assert/strict';
import {
  ARENA_W,
  ARENA_X,
  BLOCK_H,
  BLOCK_SPAWN_ABOVE,
  BLOCK_W,
  CAMERA_ANCHOR_Y,
  FOCUS_CAP,
  GROUND,
  PHASES,
  SPAWN_GRID,
} from '../src/constants.js';
import { BLOCK_TYPES } from '../src/sim/blockTypes.js';
import { framesToY, naturalSpawnRate } from '../src/sim/director.js';
import { hasNoFocusSurvivalPath } from '../src/sim/reachability.js';
import { Sim } from '../src/sim/sim.js';

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

function countBy(values) {
  const counts = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function fixedBlock(sim, col, layer, type = 'wood') {
  const block = sim.blocks.spawnColumn(col, sim.camY, type, {
    y: GROUND.y - layer * BLOCK_H,
    yVel: 0,
    shade: 0,
  });
  sim.blocks.falling.splice(sim.blocks.falling.indexOf(block), 1);
  sim.blocks.fixAt(block, layer);
  return block;
}

function runStorm(seed, frames = 36000) {
  const sim = new Sim(seed, { director: false });
  const director = sim.director;
  const drops = [];
  const eventNames = [];
  sim.player.y = 1_000_000;
  sim.events.tap = (name, payload) => {
    eventNames.push(name);
    if (name === 'stormDrop') drops.push({ ...payload, phase: director.phase });
  };

  for (let frame = 1; frame <= frames; frame++) {
    sim.frame = frame;
    director.step();
  }
  return { sim, director, drops, eventNames };
}

function phaseRate(frame, id, progress) {
  const sim = new Sim(1, { director: false });
  const director = sim.director;
  const index = PHASES.findIndex((phase) => phase.id === id);
  const spec = PHASES[index];
  sim.frame = frame;
  director.elapsedFrames = frame;
  director.phaseIndex = index;
  director.phase = id;
  director.phaseFrame = Math.max(-1, Math.floor(spec.frames * progress) - 1);
  director.updatePacing();
  return director.blockRate;
}

test('committed previews become authoritative offscreen drops without recovery injection', () => {
  const { sim, director, drops, eventNames } = runStorm(40, 300);
  assert.ok(drops.length > 0, 'opening pressure produced no block');
  const first = sim.blocks.blocks[0];
  assert.equal(first.y, -sim.camY - BLOCK_SPAWN_ABOVE);
  assert.ok(first.y + first.h < -sim.camY, 'first drop was not fully above the screen');
  assert.ok(eventNames.includes('stormForecast'));
  assert.ok(eventNames.indexOf('stormForecast') < eventNames.indexOf('stormDrop'));
  assert.ok(director.forecasts.every((forecast) => forecast.frames > 0));
  assert.equal(typeof director.planWave, 'undefined');
  assert.equal(eventNames.includes('wavePlan'), false);
  assert.equal(eventNames.includes('recoveryLift'), false);
  assert.equal(sim.blocks.blocks.some((block) => block.recovery === true), false);
});

test('an offscreen spawn moves upward when terrain rises during its forecast', () => {
  const sim = new Sim(401, { director: false });
  const director = sim.director;
  director.commitEntry({ col: 0, type: 'wood', w: BLOCK_W });
  const forecast = director.forecasts.pop();
  for (let layer = 1; layer <= 8; layer++) fixedBlock(sim, 0, layer);

  director.spawnEntry(forecast);
  const drop = sim.blocks.falling.at(-1);
  const raisedOrigin = GROUND.y - 8 * BLOCK_H - BLOCK_SPAWN_ABOVE;
  assert.ok(drop.y <= raisedOrigin, `${drop.y} did not clear ${raisedOrigin}`);
  assert.ok(drop.y + drop.h < GROUND.y - 8 * BLOCK_H);
});

test('quarter-grid blocks form supported partial overlaps and reject edge-only support', () => {
  const sim = new Sim(41, { director: false });
  const lower = fixedBlock(sim, 0, 1);
  const upper = sim.blocks.spawnColumn(3, sim.camY, 'wood', {
    y: lower.y - BLOCK_H + 1,
    yVel: 0,
    shade: 0,
  });
  sim.blocks.update();

  const overlap = Math.min(lower.x + lower.w, upper.x + upper.w) - Math.max(lower.x, upper.x);
  assert.equal(overlap, SPAWN_GRID);
  assert.equal(upper.fixed, true, 'a 15px overlap did not support wood');
  assert.equal((upper.x - ARENA_X) % SPAWN_GRID, 0);
  assert.notEqual((upper.x - ARENA_X) % BLOCK_W, 0, 'upper block collapsed to a full column');
  assert.deepEqual(sim.blocks.surfaceLayers().slice(0, 7), [1, 1, 1, 2, 2, 2, 2]);

  const edgeSim = new Sim(42, { director: false });
  const edgeLower = fixedBlock(edgeSim, 0, 1);
  const edgeOnly = edgeSim.blocks.spawnColumn(4, edgeSim.camY, 'wood', {
    y: edgeLower.y - BLOCK_H + 1,
    yVel: 0,
    shade: 0,
  });
  edgeSim.blocks.update();
  assert.equal(edgeOnly.fixed, false, 'zero-width edge contact became support');
});

test('all positive overlap is physical support, including wide beams', () => {
  const weak = new Sim(421, { director: false });
  const weakBase = fixedBlock(weak, 0, 1);
  const balancing = weak.blocks.spawnColumn(3, weak.camY, 'beam', {
    y: weakBase.y - BLOCK_H + 1,
    yVel: 0,
    w: 90,
    shade: 0,
  });
  weak.blocks.update();
  assert.equal(balancing.fixed, true, 'beam phased through a 15px contact');

  const sound = new Sim(422, { director: false });
  const soundBase = fixedBlock(sound, 0, 1);
  const supported = sound.blocks.spawnColumn(1, sound.camY, 'beam', {
    y: soundBase.y - BLOCK_H + 1,
    yVel: 0,
    w: 90,
    shade: 0,
  });
  sound.blocks.update();
  assert.equal(supported.fixed, true, '45px beam support was rejected');
});

test('offscreen wood preserves an original-like warning-to-contact window', () => {
  const wood = {
    x: ARENA_X,
    y: -BLOCK_SPAWN_ABOVE,
    yVel: 0,
    w: BLOCK_W,
    h: BLOCK_H,
    spec: BLOCK_TYPES.get('wood'),
  };
  const frames = framesToY(wood, CAMERA_ANCHOR_Y);
  assert.ok(frames >= 48 && frames <= 56, `wood contact took ${frames} frames`);
  assert.equal(framesToY({ ...wood, y: CAMERA_ANCHOR_Y - BLOCK_H }, CAMERA_ANCHOR_Y), 0);
});

test('natural rate keeps rising and phases retain surge and release texture', () => {
  const samples = [0, 3600, 18000, 54000, 216000].map(naturalSpawnRate);
  assert.equal(samples[0], 1);
  for (let i = 1; i < samples.length; i++) {
    assert.ok(samples[i] > samples[i - 1], `rate plateaued: ${samples}`);
  }
  assert.ok(samples.at(-1) - samples.at(-2) > 0.3, `tail growth was negligible: ${samples}`);

  const frame = 18000;
  const surge = phaseRate(frame, 'surge', 0.5);
  const calm = phaseRate(frame, 'calm', 0.5);
  const releaseStart = phaseRate(frame, 'release', 0);
  const releaseEnd = phaseRate(frame, 'release', 0.99);
  const buildStart = phaseRate(frame, 'build', 0);
  const buildEnd = phaseRate(frame, 'build', 0.99);
  assert.ok(surge > releaseStart && releaseStart > releaseEnd);
  assert.ok(surge > calm);
  assert.ok(buildEnd > buildStart);
});

test('the opening advances into Build instead of a second calm period', () => {
  const sim = new Sim(47, { director: false });
  for (let i = 0; i < 360; i++) {
    sim.frame++;
    sim.director.updatePacing();
  }
  assert.equal(sim.director.phase, 'build');
});

test('Focus Aim slows the storm clock instead of granting a periodic shield', () => {
  const normal = new Sim(471);
  const focused = new Sim(471);
  for (let i = 0; i < 20; i++) normal.step();
  for (let i = 0; i < 20; i++) {
    focused.step({
      ...focused.input,
      focusPressed: i === 0,
      focusHeld: true,
      focusDirX: 1,
    });
  }
  assert.ok(normal.director.elapsedFrames >= 20);
  assert.ok(focused.director.elapsedFrames < 5);
  assert.equal(focused.player.focus, FOCUS_CAP - 1, 'Focus did not spend a charge on entry');
  assert.ok(focused.player.focusAimTimer > 0, 'Focus became an instant shield/burst');
});

test('the shuffled material bag enforces bounded material frequency', () => {
  const sim = new Sim(43, { director: false });
  const director = sim.director;
  const opening = [];
  for (let i = 0; i < 4; i++) {
    opening.push(director.peekMaterial());
    director.consumeMaterial();
    director.spawnCount++;
  }
  assert.deepEqual(opening, ['wood', 'wood', 'wood', 'wood']);

  for (let bag = 0; bag < 20; bag++) {
    const materials = [];
    for (let i = 0; i < 16; i++) {
      materials.push(director.peekMaterial());
      director.consumeMaterial();
      director.spawnCount++;
    }
    assert.deepEqual(countBy(materials), { wood: 12, gravel: 3, beam: 1 });
  }
});

test('spawned pieces stay aligned and fully inside the arena', () => {
  const { drops } = runStorm(44, 12000);
  assert.ok(drops.length >= 500, `only ${drops.length} drops were sampled`);
  for (const drop of drops) {
    assert.ok(drop.x >= ARENA_X, `left overflow at ${drop.x}`);
    assert.ok(drop.x + drop.w <= ARENA_X + ARENA_W, `right overflow at ${drop.x}`);
    assert.equal((drop.x - ARENA_X) % SPAWN_GRID, 0, `misaligned x ${drop.x}`);
  }
  assert.ok(
    drops.some((drop) => (drop.x - ARENA_X) % BLOCK_W !== 0),
    'history used only full-block columns',
  );
});

test('terrain-aware safety rejects an immediate forced crush but accepts a real route', () => {
  const trapped = new Sim(45, { director: false });
  trapped.step();
  const overhead = {
    x: 370,
    y: 220,
    w: BLOCK_W,
    h: BLOCK_H,
    xVel: 0,
    yVel: 13,
    type: 'wood',
    spec: BLOCK_TYPES.get('wood'),
    frames: 0,
  };
  assert.equal(hasNoFocusSurvivalPath(trapped, overhead), false);

  const escapable = new Sim(46, { director: false });
  escapable.step();
  assert.equal(
    hasNoFocusSurvivalPath(escapable, { ...overhead, x: ARENA_X }),
    true,
  );
  assert.equal(
    escapable.director.wouldCheckmate({ col: 24, type: 'gravel', w: BLOCK_W }),
    false,
    'nonlethal gravel was rejected as checkmate',
  );
});

test('spawn safety does not rely on terrain already marked to shatter', () => {
  const sim = new Sim(461, { director: false });
  sim.step();
  const platform = fixedBlock(sim, 24, 2, 'gravel');
  const overhead = {
    x: platform.x,
    y: 220,
    w: BLOCK_W,
    h: BLOCK_H,
    xVel: 0,
    yVel: 13,
    type: 'wood',
    spec: BLOCK_TYPES.get('wood'),
    frames: 0,
  };
  assert.equal(hasNoFocusSurvivalPath(sim, overhead), true);
  sim.blocks.markFault(platform, 'carve');
  assert.equal(hasNoFocusSurvivalPath(sim, overhead), false);
});

test('long storm histories remain broad while surge clusters more than release', () => {
  const { director, drops } = runStorm(1);
  assert.ok(drops.length >= 2500, `only ${drops.length} long-run drops`);
  assert.equal(director.rejectedSpawns, 0, 'unopposed distribution run rejected a spawn');

  const positions = countBy(drops.map((drop) => drop.x));
  const legalWoodSlots = Math.floor((ARENA_W - BLOCK_W) / SPAWN_GRID) + 1;
  assert.equal(Object.keys(positions).length, legalWoodSlots, 'storm did not cover every wood slot');
  assert.ok(Math.max(...Object.values(positions)) / drops.length < 0.05, 'one x slot dominated');

  const normalizedMean = drops.reduce(
    (sum, drop) => sum + (drop.x + drop.w / 2 - ARENA_X) / ARENA_W,
    0,
  ) / drops.length;
  assert.ok(normalizedMean > 0.44 && normalizedMean < 0.56, `x mean drifted to ${normalizedMean}`);

  const texture = new Map();
  for (let i = 1; i < drops.length; i++) {
    const phase = drops[i].phase;
    const stat = texture.get(phase) ?? { count: 0, distance: 0, close: 0 };
    const distance = Math.abs(drops[i].x - drops[i - 1].x);
    stat.count++;
    stat.distance += distance;
    if (distance <= 8 * SPAWN_GRID) stat.close++;
    texture.set(phase, stat);
  }
  const surge = texture.get('surge');
  const release = texture.get('release');
  const surgeMean = surge.distance / surge.count;
  const releaseMean = release.distance / release.count;
  const surgeClose = surge.close / surge.count;
  const releaseClose = release.close / release.count;
  assert.ok(surgeMean < releaseMean * 0.8, `${surgeMean} was not below ${releaseMean}`);
  assert.ok(surgeClose > releaseClose + 0.15, `${surgeClose} vs ${releaseClose}`);
});

process.exitCode = failures ? 1 : 0;
