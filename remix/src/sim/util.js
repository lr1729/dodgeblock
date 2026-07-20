// Pure geometry helpers used by the simulation. No Phaser, no randomness.

// Ported verbatim from classic/script.js:35 — the comparisons are inclusive
// (<=), so rects that merely touch count as overlapping. The physics
// (landing snaps, ceiling push-out) depends on this; do not "fix" it.
export function rectrect(rect1, rect2) {
  return (
    ((rect1.x <= rect2.x && rect2.x <= rect1.x + rect1.w) ||
      (rect2.x <= rect1.x && rect1.x <= rect2.x + rect2.w)) &&
    ((rect1.y <= rect2.y && rect2.y <= rect1.y + rect1.h) ||
      (rect2.y <= rect1.y && rect1.y <= rect2.y + rect2.h))
  );
}

// Focus uses strict overlap so moving along a floor or support top does not
// count as entering that surface.
export function rectrectStrict(rect1, rect2) {
  return (
    rect1.x < rect2.x + rect2.w &&
    rect1.x + rect1.w > rect2.x &&
    rect1.y < rect2.y + rect2.h &&
    rect1.y + rect1.h > rect2.y
  );
}

export function constrain(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

// Block-on-block stacking test: strict in x (unlike the inclusive rectrect),
// so edge-touching neighbors don't count as support. On the SPAWN_GRID this
// guarantees every seated block overlaps its support by >= SPAWN_GRID px.
// Player and block collisions use the faithful inclusive rectangle test.
export function stackContact(a, b) {
  return (
    a.x + a.w > b.x &&
    b.x + b.w > a.x &&
    ((a.y <= b.y && b.y <= a.y + a.h) || (b.y <= a.y && a.y <= b.y + b.h))
  );
}
