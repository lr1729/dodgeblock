import {
  ARENA_W,
  ARENA_X,
  BLOCK_H,
  GAME_H,
  GROUND,
  MOVE_SPEED,
} from '../constants.js';

// This module intentionally does not prove reachability. The storm only needs
// a cheap veto for obvious local checkmates; broader terrain problems are the
// game. These margins favor the player without steering drops into safe lanes.
const REACTION_FRAMES = 10;
const SYNC_FRAMES = 10;
const IMMEDIATE_TERRAIN_FRAMES = 18;
const CLEARANCE = 6;
const MAX_ARRIVAL_FRAMES = 180;
const HARD_WALL_HEIGHT = 120;

function overlapsVertically(a, b) {
  return a.y < b.y + b.h && a.y + a.h > b.y;
}

function landingY(sim, threat) {
  let stop = GROUND.y - (threat.h ?? BLOCK_H);
  for (const support of sim.blocks.blocks) {
    if (
      !support.fixed ||
      support.faultTimer > 0 ||
      !support.spec.canSupport ||
      threat.x >= support.x + support.w ||
      threat.x + threat.w <= support.x
    ) {
      continue;
    }
    stop = Math.min(stop, support.y - (threat.h ?? BLOCK_H));
  }
  return stop;
}

function arrivalAtPlayer(sim, source, delay = 0) {
  const p = sim.player;
  const stopY = landingY(sim, source);
  if (stopY + (source.h ?? BLOCK_H) <= p.y) return Infinity;

  let y = source.y;
  let velocity = source.yVel ?? 0;
  let frames = Math.max(0, Math.ceil(delay));
  for (; frames <= MAX_ARRIVAL_FRAMES; frames++) {
    if (y + (source.h ?? BLOCK_H) >= p.y) return frames;
    velocity = Math.min(source.spec.maxFallSpeed, velocity + source.spec.gravity);
    y += velocity;
    if (y >= stopY && stopY + (source.h ?? BLOCK_H) <= p.y) return Infinity;
  }
  return Infinity;
}

function unsafeInterval(threat, playerWidth) {
  return {
    lo: threat.x - playerWidth - CLEARANCE,
    hi: threat.x + threat.w + CLEARANCE,
  };
}

function mergeIntervals(intervals) {
  if (!intervals.length) return [];
  intervals.sort((a, b) => a.lo - b.lo || a.hi - b.hi);
  const merged = [{ ...intervals[0] }];
  for (const interval of intervals.slice(1)) {
    const tail = merged[merged.length - 1];
    if (interval.lo > tail.hi) merged.push({ ...interval });
    else tail.hi = Math.max(tail.hi, interval.hi);
  }
  return merged;
}

function reachableRange(sim, arrival) {
  const p = sim.player;
  const controlledFrames = Math.max(0, arrival - REACTION_FRAMES);
  const carried = Math.abs(p.xVel ?? 0) * Math.min(arrival, 4);
  const reach = carried + MOVE_SPEED * controlledFrames + 2;
  let lo = Math.max(ARENA_X, p.x - reach);
  let hi = Math.min(ARENA_X + ARENA_W - p.w, p.x + reach);

  // At very short notice a block already beside the player is a real wall.
  // After that, jumping or wall-jumping may clear it, so the heuristic stops
  // making assumptions about terrain and leaves the random problem intact.
  const fixed = sim.blocks.blocks.filter(
    (block) => block.fixed && block.faultTimer <= 0,
  );
  const isHardWall = (root) => {
    let top = root.y;
    let bottom = root.y + root.h;
    let changed = true;
    while (changed) {
      changed = false;
      for (const block of fixed) {
        const sharedWidth = Math.min(root.x + root.w, block.x + block.w) -
          Math.max(root.x, block.x);
        if (sharedWidth < p.w || block.y > bottom + 0.1 || block.y + block.h < top - 0.1) {
          continue;
        }
        const nextTop = Math.min(top, block.y);
        const nextBottom = Math.max(bottom, block.y + block.h);
        if (nextTop !== top || nextBottom !== bottom) {
          top = nextTop;
          bottom = nextBottom;
          changed = true;
        }
      }
    }
    return p.y - top >= HARD_WALL_HEIGHT;
  };

  for (const block of fixed) {
    if (!overlapsVertically(p, block)) continue;
    if (arrival > IMMEDIATE_TERRAIN_FRAMES && !isHardWall(block)) continue;
    const interval = unsafeInterval(block, p.w);
    if (interval.hi <= p.x) lo = Math.max(lo, interval.hi);
    else if (interval.lo >= p.x) hi = Math.min(hi, interval.lo);
  }
  return { lo, hi };
}

function intervalCoversRange(intervals, range) {
  if (range.lo > range.hi) return true;
  for (const interval of mergeIntervals(intervals)) {
    if (interval.lo <= range.lo && interval.hi >= range.hi) return true;
  }
  return false;
}

function lethalThreats(sim, candidate, forecasts) {
  const threats = [];
  for (const block of sim.blocks.falling) {
    if (block.spec?.lethal) threats.push({ source: block, delay: 0 });
  }
  for (const forecast of forecasts) {
    if (forecast.spec?.lethal) {
      threats.push({ source: forecast, delay: forecast.frames ?? 0 });
    }
  }
  if (candidate.spec?.lethal) {
    threats.push({ source: candidate, delay: candidate.frames ?? 0, candidate: true });
  }
  return threats;
}

export function isObviousLocalCheckmate(sim, candidate, forecasts = []) {
  const p = sim.player;
  if (p.y > GROUND.y + GAME_H || sim.dead || !candidate.spec?.lethal) return false;

  const candidateArrival = arrivalAtPlayer(sim, candidate, candidate.frames ?? 0);
  if (!Number.isFinite(candidateArrival)) return false;
  const range = reachableRange(sim, candidateArrival);
  const candidateInterval = unsafeInterval(candidate, p.w);
  if (candidateInterval.hi < range.lo || candidateInterval.lo > range.hi) return false;

  const intervals = [];
  for (const threat of lethalThreats(sim, candidate, forecasts)) {
    const arrival = arrivalAtPlayer(sim, threat.source, threat.delay);
    if (!Number.isFinite(arrival)) continue;
    if (Math.abs(arrival - candidateArrival) > SYNC_FRAMES) continue;
    intervals.push(unsafeInterval(threat.source, p.w));
  }
  return intervalCoversRange(intervals, range);
}

export function hasNoFocusSurvivalPath(sim, candidate, forecasts = []) {
  return !isObviousLocalCheckmate(sim, candidate, forecasts);
}

// Retained for callers that displayed the old search horizon. It is now only
// an observation limit for the local arrival estimate, not a planning depth.
export const REACHABILITY_HORIZON_FRAMES = MAX_ARRIVAL_FRAMES;
