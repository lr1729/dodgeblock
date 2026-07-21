// Render-side block art: per-type vector art, the baked sprite atlas, and
// the warning strip. Sim code never imports this file.

import {
  BLOCK_W,
  BLOCK_H,
  COLOR_BLOCK_FILLS,
  COLOR_BLOCK_BORDER,
  COLOR_BLOCK_TOP,
  COLOR_BLOCK_SHADE,
  COLOR_GRAVEL,
  COLOR_BEAM,
  COLOR_WARNING,
  RES,
} from '../constants.js';
import { BLOCK_TYPES } from '../sim/blockTypes.js';

// Art definition per block type id. `fills` length must be >= the sim spec's
// `variants`. Any future material gets another row here.
export const BLOCK_ART = {
  wood: {
    fills: COLOR_BLOCK_FILLS,
    border: COLOR_BLOCK_BORDER,
    top: COLOR_BLOCK_TOP,
    shade: COLOR_BLOCK_SHADE,
  },
  gravel: {
    fills: [COLOR_GRAVEL, 0x7c817c],
    border: 0x515954,
    top: 0xc9cfca,
    shade: 0x5f6661,
    cracks: true,
  },
  beam: {
    fills: [COLOR_BEAM],
    border: 0x25313c,
    top: 0x8394a3,
    shade: 0x26333f,
    rivets: true,
  },
};

// Draws one block's vector art at scale k with its top-left at (ox, oy).
// The body rect is inset by the stroke's half-width so a texture bake at
// exactly (BLOCK_W*k, BLOCK_H*k) doesn't clip the border.
function drawBlockArt(gfx, ox, oy, art, variant, k) {
  const w = BLOCK_W * k;
  const h = BLOCK_H * k;
  gfx.lineStyle(2 * k, art.border);
  gfx.fillStyle(art.fills[variant] ?? art.fills[0]);
  gfx.fillRoundedRect(ox + k, oy + k, w - 2 * k, h - 2 * k, 6 * k);
  gfx.strokeRoundedRect(ox + k, oy + k, w - 2 * k, h - 2 * k, 6 * k);
  // bevel: light top edge, shaded bottom edge
  gfx.fillStyle(art.top, 0.55);
  gfx.fillRoundedRect(ox + 3 * k, oy + 3 * k, w - 6 * k, 8 * k, {
    tl: 4 * k, tr: 4 * k, bl: 2 * k, br: 2 * k,
  });
  gfx.fillStyle(art.shade, 0.35);
  gfx.fillRoundedRect(ox + 3 * k, oy + h - 9 * k, w - 6 * k, 6 * k, {
    tl: 2 * k, tr: 2 * k, bl: 4 * k, br: 4 * k,
  });
  if (art.cracks) {
    gfx.lineStyle(1.6 * k, art.border, 0.75);
    gfx.lineBetween(ox + 18 * k, oy + 4 * k, ox + 25 * k, oy + 16 * k);
    gfx.lineBetween(ox + 25 * k, oy + 16 * k, ox + 20 * k, oy + 26 * k);
    gfx.lineBetween(ox + 25 * k, oy + 16 * k, ox + 34 * k, oy + 21 * k);
    gfx.lineBetween(ox + 42 * k, oy + 10 * k, ox + 37 * k, oy + 28 * k);
  }
  if (art.rivets) {
    gfx.fillStyle(0xc3d0da, 0.9);
    gfx.fillCircle(ox + 10 * k, oy + 20 * k, 2.4 * k);
    gfx.fillCircle(ox + 50 * k, oy + 20 * k, 2.4 * k);
  }
}

// Immediate-mode vector draw, used by the menu's decorative blocks
export function drawBlock(gfx, b) {
  drawBlockArt(gfx, Math.round(b.x), b.y, BLOCK_ART.wood, b.shade ?? 0, 1);
}

// In-game blocks render as batched sprites: every type x variant is baked
// once into a single texture (one atlas -> one draw call for every block on
// screen), at RES density so they stay crisp under the camera zoom.
// Grid layout: one row per type, one column per variant. Frame key
// `${type}/${variant}` — see frameNameFor().
export const BLOCK_TEX = 'block-tiles';

export function frameNameFor(b) {
  const v = b.spec.frameFor ? b.spec.frameFor(b) : b.shade;
  return b.type + '/' + v;
}

export function bakeBlockTextures(scene) {
  if (scene.textures.exists(BLOCK_TEX)) return;
  const types = [...BLOCK_TYPES.values()];
  let cols = 0;
  for (const spec of types) cols = Math.max(cols, spec.variants);

  const fw = BLOCK_W * RES;
  const fh = BLOCK_H * RES;
  // one texture keeps the whole stack a single batched draw call — if the
  // type/variant count ever overflows a safe texture size, shrink RES
  // rather than splitting the atlas
  if (cols * fw > 4096 || types.length * fh > 4096) {
    throw new Error('block atlas exceeds 4096px — reduce RES or variants');
  }

  const g = scene.make.graphics({ add: false });
  for (let row = 0; row < types.length; row++) {
    const art = BLOCK_ART[types[row].id] ?? BLOCK_ART.wood;
    for (let v = 0; v < types[row].variants; v++) {
      drawBlockArt(g, v * fw, row * fh, art, v, RES);
    }
  }
  g.generateTexture(BLOCK_TEX, cols * fw, types.length * fh);
  g.destroy();
  const tex = scene.textures.get(BLOCK_TEX);
  for (let row = 0; row < types.length; row++) {
    for (let v = 0; v < types[row].variants; v++) {
      tex.add(types[row].id + '/' + v, 0, v * fw, row * fh, fw, fh);
    }
  }
}

// Pulsing warning marker for a falling block above the readable playfield.
// `y` is screen-space within the target graphics layer; `pulse` is 0..1.
export function drawWarningStrip(gfx, b, y, pulse, color = COLOR_WARNING) {
  const x = Math.round(b.x);
  const cx = x + b.w / 2;
  const h = 9;
  const art = BLOCK_ART[b.type] ?? BLOCK_ART.wood;
  gfx.fillStyle(art.fills[0], 0.45 + 0.4 * pulse);
  gfx.fillRect(x, y, b.w, h);
  gfx.lineStyle(2, color, 0.35 + 0.6 * pulse);
  gfx.strokeRect(x, y, b.w, h);
  if (b.type === 'beam') {
    gfx.fillStyle(0xd7e0e7, 0.9);
    gfx.fillCircle(x + 11, y + h / 2, 1.5);
    gfx.fillCircle(x + b.w - 11, y + h / 2, 1.5);
  } else if (b.type === 'gravel') {
    gfx.lineStyle(1.5, art.border, 0.9);
    gfx.lineBetween(cx - 7, y + 1, cx - 2, y + 5);
    gfx.lineBetween(cx - 2, y + 5, cx + 4, y + 2);
  } else {
    gfx.lineStyle(1.5, art.border, 0.72);
    gfx.lineBetween(x + 9, y + h / 2, x + b.w - 9, y + h / 2);
  }
  gfx.fillStyle(color, 0.25 + 0.65 * pulse);
  gfx.fillTriangle(cx - 7, y + h, cx + 7, y + h, cx, y + h + 3 + 5 * pulse);
}
