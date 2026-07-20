// Player rendering, extracted from the remake's Player.draw so the sim class
// stays Phaser-free. Squash & stretch and the face grid are render-only —
// collisions always use the true 30x30 box.

import {
  COLOR_PLAYER,
  COLOR_PLAYER_BORDER,
  COLOR_PLAYER_MOUTH,
} from '../constants.js';

export function drawPlayer(gfx, p, tick, opts = {}) {
  // Stretch along the committed Focus direction; otherwise keep the remake's
  // jump and landing squash so motion reads without changing the collision box.
  let stretch;
  if (p.focusTimer > 0) {
    stretch = Math.abs(p.focusDY) > Math.abs(p.focusDX) ? 0.34 : -0.34;
  } else if (p.landSquash > 0) {
    stretch = -0.16 * (p.landSquash / 8);
  } else if (p.offGround > 2) {
    stretch = Math.min(Math.abs(p.yVel) * 0.016, 0.2);
  } else {
    stretch = 0;
  }

  const w = p.w * (1 - stretch);
  const h = p.h * (1 + stretch);
  const x = p.x - (w - p.w) / 2; // keep centered horizontally
  const y = p.y + (p.h - h); // keep feet anchored

  const bodyAlpha = opts.alpha ?? 1;
  gfx.lineStyle(2, opts.border ?? COLOR_PLAYER_BORDER, bodyAlpha);
  gfx.fillStyle(opts.body ?? COLOR_PLAYER, bodyAlpha);
  gfx.fillRoundedRect(x, y, w, h, w / 6);
  gfx.strokeRoundedRect(x, y, w, h, w / 6);
  if (opts.ghost) return; // silhouette only (afterimages, best-run ghost)

  // face: 3 vertical states x 3 look directions (original costume grid)
  let eyeY, mouthY, mouthH, pupilDy;
  if (p.yVel > 0.5) {
    eyeY = 0.26;
    mouthY = 0.56;
    mouthH = 0.28; // surprised, mid-jump
    pupilDy = -1;
  } else if (p.yVel < -3.3) {
    eyeY = 0.44;
    mouthY = 0.74;
    mouthH = 0.24; // worried, falling fast
    pupilDy = 1;
  } else {
    eyeY = 0.36;
    mouthY = 0.66;
    mouthH = 0.15;
    pupilDy = 0;
  }
  let eye1X, eye2X, mouthX, pupilDx;
  if (p.xVel > 0.8) {
    eye1X = 0.34;
    eye2X = 0.76;
    mouthX = 0.42;
    pupilDx = 1;
  } else if (p.xVel < -0.8) {
    eye1X = 0.24;
    eye2X = 0.66;
    mouthX = 0.26;
    pupilDx = -1;
  } else {
    eye1X = 0.29;
    eye2X = 0.71;
    mouthX = 0.34;
    pupilDx = 0;
  }

  const eyeR = w * 0.125;
  for (const ex of [eye1X, eye2X]) {
    const cx = x + w * ex;
    const cy = y + h * eyeY;
    gfx.fillStyle(0xffffff);
    gfx.fillCircle(cx, cy, eyeR);
    gfx.fillStyle(0x1c1c1c);
    gfx.fillCircle(cx + pupilDx * eyeR * 0.35, cy + pupilDy * eyeR * 0.3, eyeR * 0.5);
  }
  gfx.fillStyle(opts.mouth ?? COLOR_PLAYER_MOUTH);
  gfx.fillRoundedRect(x + w * mouthX, y + h * mouthY, w * 0.32, h * mouthH, 3);

}
