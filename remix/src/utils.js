import Phaser from 'phaser';
import { GAME_W, GAME_H, RES, FONT } from './constants.js';

// Phaser-facing helpers only — pure geometry helpers live in sim/util.js.

// Zoom the scene camera so game code works in 800x500 coordinates while the
// canvas renders at RES times that (see RES in constants.js).
export function setupCamera(scene, bgColor) {
  const cam = scene.cameras.main;
  cam.setBackgroundColor(bgColor);
  cam.setZoom(RES);
  cam.centerOn(GAME_W / 2, GAME_H / 2);
}

// Shared text style; `resolution` makes Phaser rasterize the glyphs at canvas
// density — without it text stays blurry even on a high-res canvas.
export function textStyle(size, extra = {}) {
  return {
    fontFamily: FONT,
    fontSize: `${size}px`,
    color: '#000',
    resolution: RES,
    ...extra,
  };
}

// Digits 0-9 baked once into a RetroFont in the HUD style (white bold, dark
// stroke) at RES density. HUD numbers render as BitmapText — batched quads,
// zero rasterization when the value changes, and texel-perfect under the
// camera zoom. (A changing Phaser.Text re-rasterizes and re-uploads its
// texture on every setText, which the height display updates frequently
// and the FPS readout does every frame precisely when FPS is unstable.)
export const DIGIT_FONT = 'hud-digits';
const HUD_FONT_SIZE = 25;

export function bakeDigitFont(scene) {
  const stroke = 5 * RES;
  const pad = Math.ceil(stroke / 2) + RES; // room for the stroke overhang
  // cells are padded for the stroke; pull the advance back in so digit
  // spacing matches the padless Text look
  const spacing = { letterSpacing: -pad * 2 + RES };
  if (scene.cache.bitmapFont.exists(DIGIT_FONT)) return spacing;
  const chars = '0123456789.';
  const px = HUD_FONT_SIZE * RES;
  const font = `bold ${px}px ${FONT}`;

  const meas = document.createElement('canvas').getContext('2d');
  meas.font = font;
  let cw = 0;
  for (const c of chars) cw = Math.max(cw, Math.ceil(meas.measureText(c).width));
  cw += pad * 2;
  const ch = Math.ceil(px * 1.25) + pad * 2;

  const tex = scene.textures.createCanvas(DIGIT_FONT + '-tex', cw * chars.length, ch);
  const ctx = tex.getContext();
  ctx.font = font;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.lineWidth = stroke;
  ctx.strokeStyle = '#2b5876';
  ctx.fillStyle = '#ffffff';
  for (let i = 0; i < chars.length; i++) {
    ctx.strokeText(chars[i], i * cw + cw / 2, ch / 2);
    ctx.fillText(chars[i], i * cw + cw / 2, ch / 2);
  }
  tex.refresh();

  scene.cache.bitmapFont.add(
    DIGIT_FONT,
    Phaser.GameObjects.RetroFont.Parse(scene, {
      image: DIGIT_FONT + '-tex',
      width: cw,
      height: ch,
      chars,
      charsPerRow: chars.length,
      'spacing.x': 0,
      'spacing.y': 0,
    }),
  );
  return spacing;
}
