import {
  BLOCK_H,
  CAMERA_ANCHOR_Y,
  CAMERA_RISE_BASE,
  CAMERA_RATE_DIVISOR,
  CORNER_CORRECTION_PX,
  COYOTE_FRAMES,
  FOCUS_AIM_WORLD_SCALE,
  FOCUS_AIM_MAX_FRAMES,
  FOCUS_CAP,
  FOCUS_DASH_WORLD_SCALE,
  FOCUS_EXIT_SPEED,
  FOCUS_FRAMES,
  FOCUS_RECHARGE_LAYERS,
  FOCUS_SPEED,
  GAME_H,
  GROUND,
  JUMP_BUFFER_FRAMES,
  PLAYER_MAX_X,
  PLAYER_MIN_X,
  SQUISH_VEL,
} from '../constants.js';
import { BlockManager } from './blocks.js';
import { Director } from './director.js';
import { createEmitter } from './events.js';
import { Player } from './player.js';
import { Rng } from './rng.js';
import { constrain, rectrect, rectrectStrict } from './util.js';

export const NEUTRAL_INPUT = Object.freeze({
  up: false,
  down: false,
  left: false,
  right: false,
  jumpPressed: false,
  focusPressed: false,
  focusReleased: false,
  focusHeld: false,
  focusDirX: 0,
  focusDirY: 0,
});

export class Sim {
  constructor(seed, { director = true } = {}) {
    this.seed = seed >>> 0;
    this.rng = new Rng(this.seed);
    this.events = createEmitter();
    this.player = new Player();
    this.blocks = new BlockManager(this);
    this.director = new Director(this);
    this.directorEnabled = director;

    this.frame = 0;
    this.camY = 0;
    this.height = 0;
    this.blockRate = 0;
    this.jumpBuffer = 0;
    this.dead = false;
    this.deathCause = null;
  }

  kill(cause) {
    if (this.dead) return;
    this.dead = true;
    this.deathCause = cause;
  }

  snapshot() {
    const { supportBlock, ...player } = this.player;
    return {
      seed: this.seed,
      rngState: this.rng.s,
      frame: this.frame,
      camY: this.camY,
      height: this.height,
      blockRate: this.blockRate,
      jumpBuffer: this.jumpBuffer,
      player: {
        ...player,
        supportBlockId: supportBlock?.id ?? null,
      },
      blocks: this.blocks.snapshot(),
      director: this.director.snapshot(),
    };
  }

  restore(state) {
    this.seed = state.seed >>> 0;
    this.rng.s = state.rngState >>> 0;
    this.frame = state.frame;
    this.camY = state.camY;
    this.height = state.height;
    this.blockRate = state.blockRate;
    this.jumpBuffer = state.jumpBuffer;
    this.dead = false;
    this.deathCause = null;
    const byId = this.blocks.restore(state.blocks);
    const { supportBlockId, ...player } = state.player;
    Object.assign(this.player, player);
    this.player.supportBlock = byId.get(supportBlockId) ?? null;
    this.director.restore(state.director);
    return this;
  }

  step(inp = NEUTRAL_INPUT) {
    if (this.dead) return;
    this.frame++;
    const p = this.player;
    const wasAirborne = p.offGround > COYOTE_FRAMES;

    if (inp.jumpPressed) this.jumpBuffer = JUMP_BUFFER_FRAMES;
    let startedFocusAim = false;
    if (inp.focusPressed && p.focusTimer <= 0 && p.focusAimTimer <= 0 && p.focus > 0) {
      this.startFocusAim(inp);
      startedFocusAim = true;
    }

    if (p.focusAimTimer > 0) {
      this.updateFocusDirection(inp);
      if (!startedFocusAim) p.focusAimTimer++;
      p.focusAimRemaining = Math.max(0, p.focusAimRemaining - 1);
      if (inp.focusReleased || !inp.focusHeld || p.focusAimRemaining <= 0) {
        this.commitFocus();
      }
    }

    const worldScale = p.focusAimTimer > 0
      ? FOCUS_AIM_WORLD_SCALE
      : p.focusTimer > 0
        ? FOCUS_DASH_WORLD_SCALE
        : 1;
    if (this.directorEnabled) this.director.step(worldScale);
    this.blockRate = this.director.blockRate ?? 0;
    p.offGround += worldScale;
    p.timeSinceJump += worldScale;

    if (p.focusTimer <= 0) {
      const axis = (inp.right ? 1 : 0) - (inp.left ? 1 : 0);
      p.move(axis, worldScale);

      // Held jump repeats as soon as valid footing returns. The explicit
      // buffer still preserves short taps made just before landing.
      if ((this.jumpBuffer > 0 || inp.up) && p.jump()) {
        this.jumpBuffer = 0;
        this.events.emit('jump', { x: p.x + p.w / 2, y: p.y });
      }
    }

    if (this.jumpBuffer > 0) this.jumpBuffer = Math.max(0, this.jumpBuffer - worldScale);
    this.blocks.update(worldScale);
    if (p.focusTimer > 0) {
      this.updateFocus();
    } else {
      p.wallSide = 0;
      this.landOrSquishPass(0);
      p.updateX(worldScale);
      this.wallPass();
      p.updateWallCoyote(worldScale);
      p.updateY(inp.up, inp.down, worldScale);
      if (rectrect(p, GROUND)) this.landOnGround();
      this.landOrSquishPass(-0.1);
      p.clampX();
    }

    if (p.landSquash > 0) p.landSquash--;
    if (p.offGround === 0 && wasAirborne) {
      p.landSquash = 7;
      this.events.emit('land', { x: p.x + p.w / 2, y: p.y + p.h });
    }

    this.updateFocusRecharge(worldScale);
    this.updateCameraAndHeight(worldScale);

    if (p.y > -this.camY + GAME_H) this.kill('fell');
    if (this.dead) this.events.emit('death', { cause: this.deathCause });
  }

  directionFromInput(inp) {
    const p = this.player;
    const hasGestureDirection = inp.focusDirX !== 0 || inp.focusDirY !== 0;
    let dx = hasGestureDirection
      ? inp.focusDirX
      : inp.right
        ? 1
        : inp.left
          ? -1
          : 0;
    let dy = hasGestureDirection
      ? inp.focusDirY
      : inp.down
        ? 1
        : inp.up
          ? -1
          : 0;
    if (dx === 0 && dy === 0) dx = p.focusDX || p.facing;
    if (dx !== 0 && dy !== 0) {
      dx *= Math.SQRT1_2;
      dy *= Math.SQRT1_2;
    }
    return { dx, dy };
  }

  startFocusAim(inp) {
    const p = this.player;
    const { dx, dy } = this.directionFromInput(inp);
    p.focus--;
    p.focusAimTimer = 1;
    p.focusAimRemaining = FOCUS_AIM_MAX_FRAMES;
    p.focusDX = dx;
    p.focusDY = dy;
    this.events.emit('focusAimStart', { x: p.x, y: p.y, focus: p.focus });
  }

  updateFocusDirection(inp) {
    const { dx, dy } = this.directionFromInput(inp);
    this.player.focusDX = dx;
    this.player.focusDY = dy;
  }

  commitFocus() {
    const p = this.player;
    if (p.focusAimTimer <= 0) return;
    const heldFrames = p.focusAimTimer;
    p.focusAimTimer = 0;
    p.focusAimRemaining = 0;
    p.focusTimer = FOCUS_FRAMES;
    p.xVel = 0;
    p.yVel = 0;
    this.events.emit('focusAimEnd', { heldFrames });
    this.events.emit('focusStart', {
      x: p.x,
      y: p.y,
      dx: p.focusDX,
      dy: p.focusDY,
      focus: p.focus,
    });
  }

  updateFocus() {
    const p = this.player;
    const oldX = p.x;
    const oldY = p.y;
    const wantedX = p.x + p.focusDX * FOCUS_SPEED;
    p.x = constrain(
      wantedX,
      PLAYER_MIN_X,
      PLAYER_MAX_X - p.w,
    );
    p.y += p.focusDY * FOCUS_SPEED;

    if (p.x !== wantedX) {
      this.events.emit('focusBonk', { x: p.x + p.w / 2, y: p.y + p.h / 2 });
      this.endFocus('bonk');
      return;
    }

    const blocks = this.blocks.blocks;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (!rectrectStrict(p, b)) continue;

      if (!b.fixed) {
        this.blocks.remove(b);
        p.x = oldX;
        p.y = oldY;
        this.events.emit('focusKick', {
          x: b.x + b.w / 2,
          y: b.y + b.h / 2,
          block: b,
          shattered: true,
        });
        this.endFocus('kick');
        return;
      }

      p.x = oldX;
      p.y = oldY;
      const affected = this.blocks.removeFixedAndCollapse(b);
      this.events.emit('focusBreak', {
        x: b.x + b.w / 2,
        y: b.y + b.h / 2,
        block: b,
        fixed: true,
        affected,
        detached: affected,
      });
      this.endFocus('break');
      return;
    }

    if (rectrectStrict(p, GROUND)) {
      p.x = oldX;
      p.y = Math.min(oldY, GROUND.y - p.h);
      this.events.emit('focusBonk', { x: p.x + p.w / 2, y: GROUND.y });
      this.endFocus('bonk');
      return;
    }

    p.focusTimer--;
    if (p.focusTimer <= 0) this.endFocus('end');
  }

  focusPathCollision() {
    const p = this.player;
    const probe = { x: p.x, y: p.y, w: p.w, h: p.h };

    for (let step = 0; step < FOCUS_FRAMES; step++) {
      const wantedX = probe.x + p.focusDX * FOCUS_SPEED;
      probe.x = constrain(wantedX, PLAYER_MIN_X, PLAYER_MAX_X - probe.w);
      probe.y += p.focusDY * FOCUS_SPEED;
      const distance = (step + 1) * FOCUS_SPEED;
      if (probe.x !== wantedX) return { kind: 'bonk', reason: 'rail', distance };

      for (const block of this.blocks.blocks) {
        if (!rectrectStrict(probe, block)) continue;
        return { kind: block.fixed ? 'fixed' : 'falling', block, distance };
      }
      if (rectrectStrict(probe, GROUND)) {
        return { kind: 'bonk', reason: 'ground', distance };
      }
    }

    return { kind: 'clear' };
  }

  endFocus(reason) {
    const p = this.player;
    p.focusTimer = 0;
    p.xVel = p.focusDX * FOCUS_EXIT_SPEED;
    p.yVel = -p.focusDY * FOCUS_EXIT_SPEED;
    p.offGround = COYOTE_FRAMES + 1;
    this.events.emit('focusEnd', { reason, x: p.x, y: p.y });
  }

  landOnGround() {
    const p = this.player;
    p.y = GROUND.y - p.h;
    p.yVel = 0;
    p.offGround = 0;
    p.supportBlock = null;
    p.clearWallCoyote();
    p.lastWallJumpSide = 0;
  }

  landOrSquishPass(offset) {
    const p = this.player;
    const blocks = this.blocks.blocks;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (!rectrect(p, b)) continue;

      const contact = this.contactDirection(b);
      if (contact === 'up') {
        p.y = b.y - p.h + (b.fixed ? offset : 0);
        p.yVel = b.fixed ? 0 : -b.yVel;
        p.offGround = 0;
        p.supportBlock = b;
        p.lastWallJumpSide = 0;
        p.clearWallCoyote();
        continue;
      }

      if (
        contact === 'down' &&
        b.fixed &&
        p.yVel > 0 &&
        this.tryUpwardCornerCorrection(b)
      ) {
        continue;
      }

      if (
        contact === 'down' &&
        (!b.fixed || b.fixedAtFrame === this.frame) &&
        b.spec.lethal &&
        (b.fixed ? b.impactVel : b.yVel) > SQUISH_VEL
      ) {
        this.kill('squished');
        return;
      }

      this.resolveSoftOverlap(b, contact);
    }
  }

  contactDirection(b) {
    const p = this.player;
    const left = p.x + p.w - b.x;
    const right = b.x + b.w - p.x;
    const up = p.y + p.h - b.y;
    const down = b.y + b.h - p.y;
    const horizontal = Math.min(left, right);
    const vertical = Math.min(up, down);
    // Horizontal wins corner ties, making a glancing shoulder contact a push
    // rather than an arbitrary crush.
    if (horizontal <= vertical) return left <= right ? 'left' : 'right';
    return up <= down ? 'up' : 'down';
  }

  resolveSoftOverlap(b, contact = this.contactDirection(b)) {
    const p = this.player;
    if (contact === 'left') {
      p.x = b.x - p.w - 0.1;
      p.xVel = 0;
    } else if (contact === 'right') {
      p.x = b.x + b.w + 0.1;
      p.xVel = 0;
    } else if (contact === 'up') {
      p.y = b.y - p.h - 0.1;
      if (p.yVel < 0) p.yVel = 0;
    } else {
      p.y = b.y + b.h + 0.1;
      if (p.yVel > 0) p.yVel = 0;
    }
  }

  wallPass() {
    const p = this.player;
    const blocks = this.blocks.blocks;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (!b.fixed || !rectrect(p, b)) continue;
      const side = p.x + p.w / 2 <= b.x + b.w / 2 ? 1 : -1;
      p.x = p.originalPos;
      p.xVel = 0;
      p.rememberWall(side);
      return;
    }
  }

  tryUpwardCornerCorrection(block) {
    const p = this.player;
    const candidates = [
      { x: block.x - p.w - 0.1, direction: -1 },
      { x: block.x + block.w + 0.1, direction: 1 },
    ]
      .map((candidate) => ({
        ...candidate,
        distance: Math.abs(candidate.x - p.x),
        preference: candidate.direction === Math.sign(p.xVel) ? 0 : 1,
      }))
      .filter((candidate) => candidate.distance <= CORNER_CORRECTION_PX + 0.1)
      .sort((a, b) => a.distance - b.distance || a.preference - b.preference);

    for (const candidate of candidates) {
      if (candidate.x < PLAYER_MIN_X || candidate.x > PLAYER_MAX_X - p.w) continue;
      const probe = { x: candidate.x, y: p.y, w: p.w, h: p.h };
      const obstructed = this.blocks.blocks.some(
        (other) => other.fixed && other !== block && rectrectStrict(probe, other),
      );
      if (obstructed) continue;
      p.x = candidate.x;
      return true;
    }
    return false;
  }

  updateFocusRecharge(timeScale = 1) {
    const p = this.player;
    if (p.focusAimTimer > 0) return;
    const stable =
      p.offGround === 0 &&
      (!p.supportBlock || (p.supportBlock.fixed && p.supportBlock.faultTimer <= 0));
    if (!stable) {
      p.stableFrames = 0;
      return;
    }
    p.stableFrames += timeScale;

    const layer = Math.max(0, Math.floor((GROUND.y - (p.y + p.h) + 1) / BLOCK_H));
    if (layer > p.highestStableLayer) {
      const gained = layer - p.highestStableLayer;
      p.focusProgress += gained;
      p.highestStableLayer = layer;
      this.events.emit('focusLayer', { gained, layer });
    }
    while (p.focus < FOCUS_CAP && p.focusProgress >= FOCUS_RECHARGE_LAYERS) {
      p.focus++;
      p.focusProgress -= FOCUS_RECHARGE_LAYERS;
      this.events.emit('focusRecharge', { focus: p.focus, layer });
    }
    if (p.focus >= FOCUS_CAP) {
      p.focusProgress = Math.min(p.focusProgress, FOCUS_RECHARGE_LAYERS);
    }
    p.nextFocusLayer = p.highestStableLayer +
      Math.max(0, FOCUS_RECHARGE_LAYERS - p.focusProgress);
  }

  updateCameraAndHeight(worldScale = 1) {
    const p = this.player;
    const rise = Math.max(
      CAMERA_RISE_BASE,
      (this.blockRate ?? 0) / CAMERA_RATE_DIVISOR,
    );
    this.camY += rise * worldScale;
    if (p.y + this.camY < CAMERA_ANCHOR_Y) {
      this.camY = Math.max(this.camY, -p.y + CAMERA_ANCHOR_Y);
    }
    const stableFooting =
      p.offGround === 0 &&
      (!p.supportBlock || (p.supportBlock.fixed && p.supportBlock.faultTimer <= 0));
    if (stableFooting) {
      const stableHeight = Math.max(0, Math.round(GROUND.y - (p.y + p.h)));
      this.height = Math.max(this.height, stableHeight);
    }
  }

  hash() {
    const p = this.player;
    return [
      this.frame,
      p.x.toFixed(4),
      p.y.toFixed(4),
      p.xVel.toFixed(4),
      p.yVel.toFixed(4),
      this.camY.toFixed(4),
      this.height,
      this.blocks.length,
      this.blocks.falling.length,
      p.focus,
      p.focusAimTimer,
      p.focusAimRemaining,
      p.focusProgress,
      p.nextFocusLayer,
      this.director.phase,
      this.rng.s,
    ].join('|');
  }
}
