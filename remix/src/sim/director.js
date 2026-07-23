import {
  ARENA_W,
  ARENA_X,
  ARRIVAL_CLUSTER_FRAMES,
  BLOCK_H,
  BLOCK_SPAWN_ABOVE,
  BLOCK_W,
  GROUND,
  OPENING_FRAMES,
  PHASES,
  SPAWN_ATTEMPTS,
  SPAWN_GRID,
  SPAWN_RATE_BODY,
  SPAWN_RATE_START,
  SPAWN_RATE_TAIL,
  SPAWN_RATE_TAU_SECONDS,
  TELEGRAPH_FRAMES,
} from '../constants.js';
import { ZONES, zoneIndexFor } from '../zones.js';
import { BLOCK_TYPES } from './blockTypes.js';
import { isObviousLocalCheckmate } from './reachability.js';

const MATERIAL_BAG = [
  'wood', 'wood', 'wood', 'wood', 'wood', 'wood',
  'wood', 'wood', 'wood', 'wood', 'wood', 'wood',
  'gravel', 'gravel', 'gravel', 'beam',
];
const BEAM_WIDTH = BLOCK_W + SPAWN_GRID * 2;
const MATERIAL_COUNTS = Object.freeze({
  wood: MATERIAL_BAG.filter((type) => type === 'wood').length,
  gravel: MATERIAL_BAG.filter((type) => type === 'gravel').length,
  beam: MATERIAL_BAG.filter((type) => type === 'beam').length,
});

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

export function naturalSpawnRate(frame) {
  const seconds = Math.max(0, frame) / 60;
  const body = SPAWN_RATE_BODY * (1 - Math.exp(-seconds / SPAWN_RATE_TAU_SECONDS));
  const tail = SPAWN_RATE_TAIL * Math.sqrt(seconds / 60);
  return SPAWN_RATE_START + body + tail;
}

export function framesToY(block, targetY, limit = 240) {
  let y = block.y;
  let velocity = block.yVel ?? 0;
  if (y + block.h >= targetY) return 0;
  for (let frame = 1; frame <= limit; frame++) {
    velocity = Math.min(block.spec.maxFallSpeed, velocity + block.spec.gravity);
    y += velocity;
    if (y + block.h >= targetY) return frame;
  }
  return Infinity;
}

function phaseCorrelation(phase) {
  if (phase === 'surge') return 0.58;
  if (phase === 'build') return 0.3;
  if (phase === 'release') return 0.12;
  return 0.18;
}

export class Director {
  constructor(sim) {
    this.sim = sim;
    this.zoneIndex = zoneIndexFor(sim.camY ?? 0);
    this.phaseIndex = -1;
    this.phase = 'opening';
    this.phaseFrame = 0;
    this.elapsedFrames = 0;
    this.stage = 0;
    this.pressure = 0;
    this.blockRate = SPAWN_RATE_START;
    this.spawnAccumulator = 0;
    this.spawnCount = 0;
    this.rejectedSpawns = 0;
    this.reachabilityChecks = 0;
    this.lastSpawnX = null;
    this.materialBag = [];
    this.forecasts = [];

    this.lastDrop = null;
  }

  get zone() {
    return ZONES[this.zoneIndex];
  }

  snapshot() {
    return {
      zoneIndex: this.zoneIndex,
      phaseIndex: this.phaseIndex,
      phase: this.phase,
      phaseFrame: this.phaseFrame,
      elapsedFrames: this.elapsedFrames,
      stage: this.stage,
      pressure: this.pressure,
      blockRate: this.blockRate,
      spawnAccumulator: this.spawnAccumulator,
      spawnCount: this.spawnCount,
      rejectedSpawns: this.rejectedSpawns,
      reachabilityChecks: this.reachabilityChecks,
      lastSpawnX: this.lastSpawnX,
      materialBag: [...this.materialBag],
      forecasts: this.forecasts.map(({ spec: _spec, ...forecast }) => ({ ...forecast })),
      lastDrop: this.lastDrop ? { ...this.lastDrop } : null,
    };
  }

  restore(state) {
    const { forecasts, materialBag, lastDrop, ...values } = state;
    Object.assign(this, values);
    this.materialBag = [...materialBag];
    this.forecasts = forecasts.map((forecast) => ({
      ...forecast,
      spec: BLOCK_TYPES.get(forecast.type),
    }));
    this.lastDrop = lastDrop ? { ...lastDrop } : null;
  }

  step(timeScale = 1) {
    this.updateZone();
    this.updatePacing(timeScale);
    this.advanceForecasts(timeScale);
    this.spawnFromPressure(timeScale);
  }

  advanceForecasts(timeScale = 1) {
    for (let i = 0; i < this.forecasts.length; i++) {
      const forecast = this.forecasts[i];
      forecast.frames -= timeScale;
      if (forecast.frames > 0) continue;
      this.forecasts.splice(i--, 1);
      this.spawnEntry(forecast);
    }
  }

  updateZone() {
    const next = zoneIndexFor(this.sim.camY ?? 0);
    if (next === this.zoneIndex) return;
    const direction = next > this.zoneIndex ? 1 : -1;
    while (this.zoneIndex !== next) {
      this.zoneIndex += direction;
      this.sim.events.emit('zoneChange', { index: this.zoneIndex, zone: this.zone });
    }
  }

  updatePacing(timeScale = 1) {
    this.elapsedFrames += timeScale;
    this.stage = Math.floor(Math.max(0, this.elapsedFrames) / 3600);
    this.phaseFrame += timeScale;

    if (this.phase === 'opening') {
      if (this.phaseFrame >= OPENING_FRAMES) {
        // The opening is already the quiet read-in, so enter Build instead of
        // making the player sit through a second consecutive calm period.
        this.phaseIndex = PHASES.findIndex((phase) => phase.id === 'build');
        this.phase = PHASES[this.phaseIndex].id;
        this.phaseFrame = 0;
        this.sim.events.emit('phaseChange', { phase: this.phase });
      }
    } else {
      let spec = PHASES[this.phaseIndex];
      if (this.phaseFrame >= spec.frames) {
        this.phaseIndex = (this.phaseIndex + 1) % PHASES.length;
        spec = PHASES[this.phaseIndex];
        this.phase = spec.id;
        this.phaseFrame = 0;
        this.sim.events.emit('phaseChange', { phase: this.phase });
      }
    }

    let multiplier = 1;
    if (this.phase !== 'opening') {
      const spec = PHASES[this.phaseIndex];
      const t = clamp(this.phaseFrame / spec.frames, 0, 1);
      multiplier = spec.from + (spec.to - spec.from) * t;
    }
    this.blockRate = naturalSpawnRate(this.elapsedFrames) * multiplier;
    this.pressure = clamp((this.blockRate - SPAWN_RATE_START) / 5.6, 0, 1);
  }

  spawnFromPressure(timeScale = 1) {
    this.spawnAccumulator += (this.blockRate / 60) * timeScale;
    let spawned = 0;
    while (this.spawnAccumulator >= 1 && spawned < 3) {
      const type = this.peekMaterial();
      const entry = this.findSpawn(type);
      if (!entry) {
        this.rejectedSpawns++;
        this.spawnAccumulator = Math.min(this.spawnAccumulator, 1);
        return;
      }
      this.consumeMaterial();
      this.commitEntry(entry);
      this.spawnAccumulator--;
      spawned++;
    }
  }

  peekMaterial() {
    if (this.spawnCount < 4) return 'wood';
    if (!this.materialBag.length) this.refillMaterialBag();
    return this.materialBag[0];
  }

  consumeMaterial() {
    if (this.spawnCount >= 4) this.materialBag.shift();
  }

  refillMaterialBag() {
    this.materialBag = [...MATERIAL_BAG];
    this.shuffleMaterialBag();
  }

  shuffleMaterialBag() {
    for (let i = this.materialBag.length - 1; i > 0; i--) {
      const j = this.sim.rng.int(0, i + 1);
      [this.materialBag[i], this.materialBag[j]] = [this.materialBag[j], this.materialBag[i]];
    }
  }

  materialRemainderCounts() {
    const counts = { wood: 0, gravel: 0, beam: 0 };
    if (!this.materialBag.length) return { ...MATERIAL_COUNTS };
    for (const type of this.materialBag) counts[type]++;
    return counts;
  }

  reshuffleMaterialRemainder() {
    // An empty bag is lazily refilled by peekMaterial. Leaving it empty keeps
    // the future RNG call order identical while still representing a full bag.
    if (this.materialBag.length) this.shuffleMaterialBag();
  }

  findSpawn(type) {
    const width = type === 'beam' ? BEAM_WIDTH : BLOCK_W;
    const maxCol = Math.floor((ARENA_W - width) / SPAWN_GRID);
    for (let attempt = 0; attempt < SPAWN_ATTEMPTS; attempt++) {
      let col;
      if (
        this.lastSpawnX !== null &&
        this.sim.rng.chance(phaseCorrelation(this.phase))
      ) {
        const previous = Math.round((this.lastSpawnX - ARENA_X) / SPAWN_GRID);
        col = clamp(previous + this.sim.rng.int(-8, 9), 0, maxCol);
      } else {
        col = this.sim.rng.int(0, maxCol + 1);
      }
      const entry = { col, type, w: width };
      if (!this.wouldCheckmate(entry)) return entry;
    }
    return null;
  }

  wouldCheckmate(entry) {
    const spec = BLOCK_TYPES.get(entry.type);
    if (!spec?.lethal) return false;

    const proposed = {
      x: ARENA_X + entry.col * SPAWN_GRID,
      y: this.spawnYFor(entry),
      yVel: 0,
      w: entry.w,
      h: BLOCK_H,
      spec,
      type: entry.type,
      frames: TELEGRAPH_FRAMES,
    };
    if (!this.needsReachabilityCheck(proposed)) return false;
    this.reachabilityChecks++;
    return isObviousLocalCheckmate(this.sim, proposed, this.forecasts);
  }

  needsReachabilityCheck(proposed) {
    const p = this.sim.player;
    const near = (block, margin = 90) =>
      p.x + p.w >= block.x - margin && p.x <= block.x + block.w + margin;
    if (near(proposed)) return true;

    const proposedArrival = TELEGRAPH_FRAMES + framesToY(proposed, p.y);
    const existing = [
      ...this.sim.blocks.falling.map((block) => ({ ...block, frames: 0 })),
      ...this.forecasts,
    ];
    for (const threat of existing) {
      if (!threat.spec?.lethal || !near(threat)) continue;
      const arrival = (threat.frames ?? 0) + framesToY(threat, p.y);
      if (Math.abs(arrival - proposedArrival) > ARRIVAL_CLUSTER_FRAMES) continue;
      const gap = Math.max(
        0,
        proposed.x - (threat.x + threat.w),
        threat.x - (proposed.x + proposed.w),
      );
      if (gap <= 180) return true;
    }
    return false;
  }

  spawnYFor(entry) {
    const surface = this.sim.blocks.surfaceLayers();
    const cells = Math.ceil(entry.w / SPAWN_GRID);
    let localTop = 0;
    for (let i = 0; i < cells; i++) {
      localTop = Math.max(localTop, surface[entry.col + i] ?? 0);
    }
    const aboveCamera = -this.sim.camY - BLOCK_SPAWN_ABOVE;
    const aboveTerrain = GROUND.y - localTop * BLOCK_H - BLOCK_SPAWN_ABOVE;
    return Math.min(aboveCamera, aboveTerrain);
  }

  commitEntry(entry) {
    const forecast = {
      ...entry,
      x: ARENA_X + entry.col * SPAWN_GRID,
      y: this.spawnYFor(entry),
      h: BLOCK_H,
      spec: BLOCK_TYPES.get(entry.type),
      frames: TELEGRAPH_FRAMES,
    };
    this.spawnCount++;
    this.lastSpawnX = forecast.x;
    this.forecasts.push(forecast);
    this.sim.events.emit('stormForecast', forecast);
  }

  spawnEntry(entry) {
    // The rail promises x, width, and material. Terrain may rise during the
    // forecast, so the still-offscreen origin may move only farther upward to
    // preserve valid occupancy and at least the advertised warning time.
    const opts = { y: Math.min(entry.y, this.spawnYFor(entry)), reserve: true };
    if (entry.w !== BLOCK_W) opts.w = entry.w;
    const block = this.sim.blocks.spawnColumn(
      entry.col,
      this.sim.camY,
      entry.type,
      opts,
    );
    this.lastDrop = {
      x: block.x,
      w: block.w,
      type: block.type,
      frame: this.sim.frame,
    };
    this.sim.events.emit('stormDrop', this.lastDrop);
  }
}
