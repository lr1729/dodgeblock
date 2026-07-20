// Game-feel systems: hitstop, slow-mo, screenshake, full-screen flash.
// Entirely render-side. Hitstop and slow-mo work by scaling the accumulator
// drain in GameScene.update — the fixed timestep itself is untouched, every
// executed step is still exactly STEP_MS of sim time, so determinism holds:
// juice changes WHEN steps run, never what they compute.
//
// Shake is capped and applied to the world container so the HUD stays steady.

import { STEP_MS } from '../constants.js';

export class Juice {
  constructor() {
    this.now = 0;
    this.hitstopLeft = 0;
    this.lastHitstop = -Infinity;
    this.slowLeft = 0;
    this.slowScale = 1;
    this.shakeAmp = 0;
    this.shakeLeft = 0;
    this.shakeDur = 1;
    this.shakePriority = 0;
    this.shakeAxis = 'both';
    this.leanX = 0; // Focus lean target, eased in offset()
    this.leanCur = 0;
    this.flashes = [];
  }

  // wall-clock decay; call once per render frame before draining the accumulator
  update(delta) {
    this.now += delta;
    if (this.hitstopLeft > 0) {
      this.hitstopLeft = Math.max(0, this.hitstopLeft - delta);
    } else if (this.slowLeft > 0) {
      this.slowLeft = Math.max(0, this.slowLeft - delta);
    }
    if (this.shakeLeft > 0) {
      this.shakeLeft = Math.max(0, this.shakeLeft - delta);
      if (this.shakeLeft === 0) this.shakePriority = 0;
    }
    this.leanCur += (this.leanX - this.leanCur) * Math.min(1, delta / 60);
    for (let i = 0; i < this.flashes.length; i++) {
      const f = this.flashes[i];
      f.left -= delta;
      if (f.left <= 0) this.flashes.splice(i--, 1);
    }
  }

  // multiply the accumulator input by this
  get timeScale() {
    if (this.hitstopLeft > 0) return 0;
    if (this.slowLeft > 0) return this.slowScale;
    return 1;
  }

  hitstop(frames) {
    if (this.now - this.lastHitstop < 500) return;
    this.lastHitstop = this.now;
    this.hitstopLeft = frames * STEP_MS;
  }

  // scale < 1; duration given in sim frames (wall time stretches to match)
  slowmo(scale, frames) {
    this.slowScale = scale;
    this.slowLeft = (frames * STEP_MS) / scale;
  }

  shake(amp, ms, priority = 1, axis = 'both') {
    if (this.shakeLeft > 0 && priority < this.shakePriority) return;
    this.shakeAmp = Math.min(amp, 10);
    this.shakeDur = ms;
    this.shakeLeft = ms;
    this.shakePriority = priority;
    this.shakeAxis = axis;
  }

  // World-container offset for this frame (shake plus Focus lean).
  offset() {
    let ox = this.leanCur;
    let oy = 0;
    if (this.shakeLeft > 0) {
      const a = this.shakeAmp * (this.shakeLeft / this.shakeDur);
      if (a >= 0.5) {
        if (this.shakeAxis !== 'y') ox += (Math.random() * 2 - 1) * a;
        oy += (Math.random() * 2 - 1) * a;
      }
    }
    return { ox, oy };
  }

  flash(color, alpha, ms) {
    this.flashes.push({ color, alpha, ms, left: ms });
  }

  drawFlashes(gfx, w, h) {
    gfx.clear();
    for (const f of this.flashes) {
      gfx.fillStyle(f.color, f.alpha * (f.left / f.ms));
      gfx.fillRect(0, 0, w, h);
    }
  }
}
