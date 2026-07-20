// Deterministic RNG for the simulation. One mulberry32 stream drives ALL sim
// randomness (block shades and director choices) so
// a run is a pure function of (seed, input history). Render-side code keeps
// using Math.random — cosmetic randomness must never perturb the sim stream.

export class Rng {
  constructor(seed) {
    this.s = seed >>> 0;
  }

  // mulberry32 — float in [0, 1)
  next() {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // p5-style random(min, max) floored — same shape as the remake's randomInt
  int(min, max) {
    return Math.floor(this.next() * (max - min) + min);
  }

  range(min, max) {
    return this.next() * (max - min) + min;
  }

  pick(arr) {
    return arr[Math.floor(this.next() * arr.length)];
  }

  chance(p) {
    return this.next() < p;
  }

  // weighted pick from [{ value, w }] — used by spawn/event tables
  weighted(table) {
    let total = 0;
    for (const e of table) total += e.w;
    let r = this.next() * total;
    for (const e of table) {
      if ((r -= e.w) < 0) return e.value;
    }
    return table[table.length - 1].value;
  }
}

// xmur3 string hash -> 32-bit seed for stable named/debug seeds.
export function seedFromString(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  h = Math.imul(h ^ (h >>> 16), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  return (h ^= h >>> 16) >>> 0;
}
