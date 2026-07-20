// Render-side eye candy: particles and scenery helpers. Nothing in this file
// touches simulation state — gameplay stays byte-identical with it removed.

import { GAME_W, GAME_H } from '../constants.js';

const MAX_PARTICLES = 400;

export class ParticleFx {
  constructor() {
    this.parts = [];
  }

  // soft puff, e.g. a block or the player landing
  dust(x, y, n = 6, color = 0xf0e6d2) {
    if (this.parts.length > MAX_PARTICLES) return;
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI; // fan upward-ish
      const sp = 0.6 + Math.random() * 1.2;
      const life = 18 + Math.random() * 12;
      this.parts.push({
        x: x + (Math.random() - 0.5) * 14,
        y,
        vx: Math.cos(a) * sp * (Math.random() < 0.5 ? 1 : -1),
        vy: -Math.abs(Math.sin(a)) * sp,
        grav: 0.06,
        life,
        maxLife: life,
        size: 2 + Math.random() * 2.5,
        color,
      });
    }
  }

  // Radial sparkle ring for Focus recharge and other positive feedback.
  burst(x, y, color, n = 10) {
    if (this.parts.length > MAX_PARTICLES) return;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const sp = 1.8 + Math.random() * 0.8;
      const life = 20 + Math.random() * 8;
      this.parts.push({
        x,
        y,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp,
        grav: 0,
        life,
        maxLife: life,
        size: 2.5,
        color,
      });
    }
  }

  // Rotating block-shard quads with gravity for Focus breaks.
  shards(x, y, color, n = 12) {
    if (this.parts.length > MAX_PARTICLES) return;
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const sp = 1.5 + Math.random() * 2.5;
      const life = 26 + Math.random() * 10;
      this.parts.push({
        x: x + (Math.random() - 0.5) * 20,
        y: y + (Math.random() - 0.5) * 10,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp - 2,
        grav: 0.3,
        life,
        maxLife: life,
        size: 3 + Math.random() * 4,
        color,
        shard: true,
        ang: Math.random() * Math.PI,
        spin: (Math.random() - 0.5) * 0.4,
      });
    }
  }

  update() {
    const parts = this.parts;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += p.grav;
      if (p.shard) p.ang += p.spin;
      if (--p.life <= 0) {
        parts.splice(i, 1);
        i--;
      }
    }
  }

  draw(gfx) {
    for (const p of this.parts) {
      const k = p.life / p.maxLife;
      gfx.fillStyle(p.color, k * 0.9);
      if (p.shard) {
        // rotated quad from four computed corners
        const s = p.size * (0.4 + 0.6 * k);
        const c = Math.cos(p.ang) * s;
        const n = Math.sin(p.ang) * s;
        gfx.fillTriangle(p.x - c, p.y - n, p.x + n, p.y - c, p.x + c, p.y + n);
        gfx.fillTriangle(p.x + c, p.y + n, p.x - n, p.y + c, p.x - c, p.y - n);
      } else {
        gfx.fillCircle(p.x, p.y, p.size * (0.5 + 0.5 * k));
      }
    }
  }
}

// --- scenery helpers shared by the scenes ---

export function drawSkyGradient(gfx, top, bottom) {
  gfx.fillGradientStyle(top, top, bottom, bottom, 1);
  gfx.fillRect(0, 0, GAME_W, GAME_H);
}

export function drawCloud(gfx, x, y, s, alpha = 0.75) {
  gfx.fillStyle(0xffffff, alpha);
  gfx.fillEllipse(x, y, 70 * s, 26 * s);
  gfx.fillEllipse(x - 22 * s, y + 5 * s, 44 * s, 18 * s);
  gfx.fillEllipse(x + 24 * s, y + 6 * s, 48 * s, 20 * s);
}

// A handful of clouds with random height, size, and drift speed
export function makeClouds(n = 5) {
  const clouds = [];
  for (let i = 0; i < n; i++) {
    clouds.push({
      x: Math.random() * 1040,
      y: Math.random() * 560,
      s: 0.7 + Math.random() * 0.9,
      drift: 0.08 + Math.random() * 0.12,
    });
  }
  return clouds;
}

// Screen-space parallax: clouds drift sideways with time and slide down as
// the camera climbs (world moves down when the camera moves up).
export function drawClouds(gfx, clouds, tick, camY) {
  gfx.clear();
  for (const c of clouds) {
    const sx = ((((c.x + tick * c.drift) % 1040) + 1040) % 1040) - 120;
    const sy = ((((c.y + camY * 0.15) % 560) + 560) % 560) - 30;
    drawCloud(gfx, sx, sy, c.s);
  }
}
