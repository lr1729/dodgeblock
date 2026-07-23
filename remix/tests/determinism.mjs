import assert from 'node:assert/strict';
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

// Run only the authoritative storm systems. Keeping the player far below the
// arrival horizon makes the local safety veto irrelevant, so this isolates the
// seeded pacing, material, and placement streams from controller behavior.
function stormHistory(seed, frames = 7200) {
  const sim = new Sim(seed, { director: false });
  const director = sim.director;
  const drops = [];
  sim.player.y = 1_000_000;
  sim.events.tap = (name, payload) => {
    if (name !== 'stormDrop') return;
    drops.push([
      payload.frame,
      director.phase,
      payload.type,
      payload.x,
      payload.w,
    ]);
  };

  for (let frame = 1; frame <= frames; frame++) {
    sim.frame = frame;
    director.step();
  }

  return {
    drops,
    phase: director.phase,
    accumulator: director.spawnAccumulator.toFixed(9),
    spawnCount: director.spawnCount,
    rejectedSpawns: director.rejectedSpawns,
    bag: [...director.materialBag],
    rng: sim.rng.s,
  };
}

test('same seed reproduces the complete spawn history', () => {
  for (const seed of [1, 42, 0xdecafbad]) {
    assert.deepEqual(stormHistory(seed), stormHistory(seed), `seed ${seed} diverged`);
  }
});

test('different seeds produce diverse placement and material histories', () => {
  const signatures = new Set();
  for (let seed = 1; seed <= 10; seed++) {
    const history = stormHistory(seed, 4800).drops;
    assert.ok(history.length >= 100, `seed ${seed} produced only ${history.length} drops`);
    signatures.add(JSON.stringify(history.slice(0, 80)));
  }
  assert.ok(signatures.size >= 9, `only ${signatures.size} distinct histories across ten seeds`);
});

test('reseeded material remainders preserve counts while changing hidden order', () => {
  const sim = new Sim(91, { director: false });
  sim.director.spawnCount = 4;
  sim.director.materialBag = [
    'wood', 'gravel', 'wood', 'beam', 'wood', 'gravel',
  ];
  const before = sim.director.materialRemainderCounts();
  sim.rng.s = 123456;
  sim.director.reshuffleMaterialRemainder();
  assert.deepEqual(sim.director.materialRemainderCounts(), before);
  assert.equal(sim.director.materialBag.length, 6);

  sim.director.materialBag = [];
  sim.rng.s = 987;
  sim.director.reshuffleMaterialRemainder();
  assert.deepEqual(
    sim.director.materialRemainderCounts(),
    { wood: 12, gravel: 3, beam: 1 },
    'an empty lazy bag remains a complete future bag',
  );
  assert.deepEqual(sim.director.materialBag, []);
});

process.exitCode = failures ? 1 : 0;
