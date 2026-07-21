import {
  BLOCK_H,
  CAMERA_ANCHOR_Y,
  CAMERA_RISE_BASE,
  CAMERA_RATE_DIVISOR,
  CORNER_CORRECTION_PX,
  COYOTE_FRAMES,
  DOWNWARD_CORRECTION_PX,
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
import { normalizeRunRules } from '../rules.js';
import { constrain, rectrectStrict } from './util.js';

const CONTACT_EPSILON = 0.001;

function overlapsX(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x;
}

function overlapsY(a, b) {
  return a.y < b.y + b.h && a.y + a.h > b.y;
}

function sweptRectContact(mover, dx, dy, target) {
  const axis = (start, size, delta, targetStart, targetSize) => {
    if (delta > 0) {
      return {
        entry: (targetStart - (start + size)) / delta,
        exit: (targetStart + targetSize - start) / delta,
      };
    }
    if (delta < 0) {
      return {
        entry: (targetStart + targetSize - start) / delta,
        exit: (targetStart - (start + size)) / delta,
      };
    }
    if (start >= targetStart + targetSize || start + size <= targetStart) return null;
    return { entry: -Infinity, exit: Infinity };
  };

  const x = axis(mover.x, mover.w, dx, target.x, target.w);
  const y = axis(mover.y, mover.h, dy, target.y, target.h);
  if (!x || !y) return null;
  const entry = Math.max(x.entry, y.entry);
  const exit = Math.min(x.exit, y.exit);
  if (entry > exit + CONTACT_EPSILON || exit < 0 || entry > 1) return null;
  if (entry < 0 && !rectrectStrict(mover, target)) return null;

  const time = constrain(entry, 0, 1);
  const at = {
    x: mover.x + dx * time,
    y: mover.y + dy * time,
    w: mover.w,
    h: mover.h,
  };
  const overlapAtX = Math.max(0, Math.min(at.x + at.w, target.x + target.w) - Math.max(at.x, target.x));
  const overlapAtY = Math.max(0, Math.min(at.y + at.h, target.y + target.h) - Math.max(at.y, target.y));
  const coverage = x.entry > y.entry + CONTACT_EPSILON
    ? overlapAtY
    : y.entry > x.entry + CONTACT_EPSILON
      ? overlapAtX
      : Math.max(overlapAtX, overlapAtY);
  return { time, coverage };
}

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
  constructor(seed, { director = true, rules } = {}) {
    this.seed = seed >>> 0;
    this.rng = new Rng(this.seed);
    this.events = createEmitter();
    this.player = new Player();
    this.blocks = new BlockManager(this);
    this.director = new Director(this);
    this.directorEnabled = director;
    this.rules = normalizeRunRules(rules);

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
      rules: this.rules,
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
    this.rules = normalizeRunRules(state.rules ?? this.rules);
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
    this.resolveIncomingBlockMotion();
    if (this.dead) {
      this.events.emit('death', { cause: this.deathCause });
      return;
    }
    if (p.focusTimer > 0) {
      this.updateFocus();
    } else {
      p.wallSide = 0;
      const fromX = p.x;
      p.updateX(worldScale);
      this.resolvePlayerX(fromX);
      const fromY = p.y;
      p.updateY(inp.up, inp.down, worldScale);
      this.resolvePlayerY(fromY);
      this.detectWallContact();
      p.updateWallCoyote(worldScale);
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
    if (dx === 0 && dy === 0) {
      if (p.focusAimTimer > 0 && (p.focusDX !== 0 || p.focusDY !== 0)) {
        return { dx: p.focusDX, dy: p.focusDY };
      }
      dx = p.facing;
    }
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
    p.focusCommittedFrame = -1;
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
    p.focusCommittedFrame = this.frame;
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

    const collision = this.firstFocusBlockCollision(oldX, oldY, p.x - oldX, p.y - oldY);
    if (collision) {
      const b = collision.block;
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

    if (p.x !== wantedX) {
      this.events.emit('focusBonk', { x: p.x + p.w / 2, y: p.y + p.h / 2 });
      this.endFocus('bonk');
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

  firstFocusBlockCollision(x, y, dx, dy) {
    const mover = { x, y, w: this.player.w, h: this.player.h };
    const hits = [];
    for (const block of this.blocks.blocks) {
      const contact = sweptRectContact(mover, dx, dy, block);
      if (contact) hits.push({ block, ...contact });
    }
    hits.sort((a, b) =>
      a.time - b.time ||
      b.coverage - a.coverage ||
      a.block.id - b.block.id
    );
    return hits[0] ?? null;
  }

  focusPathCollision() {
    const p = this.player;
    const probe = { x: p.x, y: p.y, w: p.w, h: p.h };

    for (let step = 0; step < FOCUS_FRAMES; step++) {
      const oldX = probe.x;
      const oldY = probe.y;
      const wantedX = probe.x + p.focusDX * FOCUS_SPEED;
      probe.x = constrain(wantedX, PLAYER_MIN_X, PLAYER_MAX_X - probe.w);
      probe.y += p.focusDY * FOCUS_SPEED;
      const stepDistance = Math.hypot(probe.x - oldX, probe.y - oldY);
      const distance = step * FOCUS_SPEED + stepDistance;
      const collision = this.firstFocusBlockCollision(
        oldX,
        oldY,
        probe.x - oldX,
        probe.y - oldY,
      );
      if (collision) {
        return {
          kind: collision.block.fixed ? 'fixed' : 'falling',
          block: collision.block,
          distance: step * FOCUS_SPEED + collision.time * stepDistance,
        };
      }
      if (probe.x !== wantedX) return { kind: 'bonk', reason: 'rail', distance };
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

  resolveIncomingBlockMotion() {
    const p = this.player;
    const moving = this.blocks.blocks
      .filter((b) => !b.fixed || b.fixedAtFrame === this.frame)
      .sort((a, b) => a.id - b.id);
    const crushes = [];

    for (const b of moving) {
      if (!overlapsX(p, b)) continue;
      const previousBottom = (b.previousY ?? b.y) + b.h;
      const currentBottom = b.y + b.h;
      const crossedHead = previousBottom <= p.y + CONTACT_EPSILON &&
        currentBottom >= p.y - CONTACT_EPSILON;
      const penetratingFromAbove = rectrectStrict(p, b) &&
        (b.previousY ?? b.y) < p.y + p.h / 2;
      const playerCenterX = p.x + p.w / 2;
      const horizontallyOverhead = playerCenterX > b.x && playerCenterX < b.x + b.w;
      const impactVelocity = b.fixed ? b.impactVel : b.yVel;
      if (
        b.spec.lethal &&
        impactVelocity > SQUISH_VEL &&
        horizontallyOverhead &&
        (crossedHead || penetratingFromAbove)
      ) {
        const travel = Math.max(CONTACT_EPSILON, currentBottom - previousBottom);
        const time = constrain((p.y - previousBottom) / travel, 0, 1);
        crushes.push({ block: b, time });
      }
    }

    if (crushes.length) {
      crushes.sort((a, b) => a.time - b.time || a.block.id - b.block.id);
      if (!this.tryAutoGuard(crushes.map(({ block }) => block))) {
        this.kill('squished');
        return;
      }
    }

    for (const b of moving) {
      if (!this.blocks.blocks.includes(b) || !rectrectStrict(p, b)) continue;
      const blockCenter = b.x + b.w / 2;
      const playerCenter = p.x + p.w / 2;
      if (p.y < (b.previousY ?? b.y) - CONTACT_EPSILON) continue;
      if (playerCenter > b.x && playerCenter < b.x + b.w && b.y < p.y) {
        const below = b.y + b.h;
        if (this.canPlacePlayer(p.x, below, b)) {
          p.y = below;
          if (p.yVel > 0) p.yVel = 0;
          continue;
        }
      }
      const sides = playerCenter <= blockCenter
        ? [b.x - p.w, b.x + b.w]
        : [b.x + b.w, b.x - p.w];
      const openSide = sides.find((x) => this.canPlacePlayer(x, p.y, b));
      if (openSide !== undefined) {
        p.x = openSide;
        p.xVel = 0;
        continue;
      }

      const above = b.y - p.h;
      if (this.canPlacePlayer(p.x, above, b)) {
        p.y = above;
        p.yVel = b.fixed ? 0 : -b.yVel;
        p.offGround = 0;
        p.supportBlock = b;
        p.lastWallJumpSide = 0;
        p.clearWallCoyote();
        continue;
      }

      const impactVelocity = b.fixed ? b.impactVel : b.yVel;
      if (b.spec.lethal && impactVelocity > SQUISH_VEL) {
        if (!this.tryAutoGuard([b])) {
          this.kill('squished');
          return;
        }
      } else {
        const below = b.y + b.h;
        if (this.canPlacePlayer(p.x, below, b)) {
          p.y = below;
          if (p.yVel > 0) p.yVel = 0;
        }
      }
    }
  }

  canPlacePlayer(x, y, ignoredBlock = null) {
    const p = this.player;
    if (x < PLAYER_MIN_X || x > PLAYER_MAX_X - p.w) return false;
    const probe = { x, y, w: p.w, h: p.h };
    return !this.blocks.blocks.some(
      (block) => block !== ignoredBlock && rectrectStrict(probe, block),
    ) && !rectrectStrict(probe, GROUND);
  }

  tryAutoGuard(blocks) {
    const p = this.player;
    if (!this.rules.autoGuard) return false;
    const usedAimCharge = p.focusAimTimer > 0 || p.focusCommittedFrame === this.frame;
    const sameIncident = p.lastGuardFrame === this.frame;
    if (!usedAimCharge && !sameIncident && p.focus <= 0) return false;

    if (!usedAimCharge && !sameIncident) p.focus--;
    if (p.focusAimTimer > 0) {
      const heldFrames = p.focusAimTimer;
      p.focusAimTimer = 0;
      p.focusAimRemaining = 0;
      this.events.emit('focusAimEnd', { heldFrames, guarded: true });
    }
    p.focusTimer = 0;
    p.focusCommittedFrame = -1;
    p.lastGuardFrame = this.frame;
    p.yVel = Math.max(1.8, p.yVel);
    const incident = new Set();
    for (const block of blocks) {
      incident.add(block);
      if (block.fixed) {
        for (const dependent of this.blocks.dependentBranch(block)) incident.add(dependent);
      }
    }
    const removed = [];
    for (const block of [...incident].sort((a, b) => a.id - b.id)) {
      if (!this.blocks.blocks.includes(block)) continue;
      removed.push({ ...block });
      this.blocks.remove(block);
    }
    this.events.emit('autoGuard', { blocks: removed, focus: p.focus, x: p.x, y: p.y });
    return true;
  }

  resolvePlayerX(fromX) {
    const p = this.player;
    const dx = p.x - fromX;
    if (Math.abs(dx) <= CONTACT_EPSILON) return;
    const candidates = [];
    for (const b of this.blocks.blocks) {
      if (!overlapsY(p, b)) continue;
      if (dx > 0 && fromX + p.w <= b.x + CONTACT_EPSILON && p.x + p.w >= b.x) {
        candidates.push({ block: b, face: b.x - p.w, time: (b.x - (fromX + p.w)) / dx });
      } else if (dx < 0 && fromX >= b.x + b.w - CONTACT_EPSILON && p.x <= b.x + b.w) {
        candidates.push({ block: b, face: b.x + b.w, time: ((b.x + b.w) - fromX) / dx });
      }
    }
    candidates.sort((a, b) => a.time - b.time || a.block.id - b.block.id);
    const hit = candidates[0];
    if (!hit) return;
    p.x = hit.face;
    p.xVel = 0;
    p.rememberWall(dx > 0 ? 1 : -1);
  }

  resolvePlayerY(fromY, corrected = false) {
    const p = this.player;
    const dy = p.y - fromY;
    if (Math.abs(dy) <= CONTACT_EPSILON) return;
    const candidates = [];

    if (dy > 0) {
      if (fromY + p.h <= GROUND.y + CONTACT_EPSILON && p.y + p.h >= GROUND.y) {
        candidates.push({ block: null, face: GROUND.y - p.h, time: (GROUND.y - (fromY + p.h)) / dy });
      }
      for (const b of this.blocks.blocks) {
        if (!overlapsX(p, b)) continue;
        const oldTop = b.fixed ? b.y : b.previousY ?? b.y;
        if (fromY + p.h <= oldTop + CONTACT_EPSILON && p.y + p.h >= b.y) {
          candidates.push({
            block: b,
            face: b.y - p.h,
            time: constrain((oldTop - (fromY + p.h)) / Math.max(CONTACT_EPSILON, dy + oldTop - b.y), 0, 1),
          });
        } else if (
          rectrectStrict(p, b) &&
          fromY + p.h / 2 <= (b.previousY ?? b.y) + b.h / 2
        ) {
          candidates.push({ block: b, face: b.y - p.h, time: 0 });
        }
      }
    } else {
      for (const b of this.blocks.blocks) {
        if (!overlapsX(p, b)) continue;
        const bottom = b.y + b.h;
        if (fromY >= bottom - CONTACT_EPSILON && p.y <= bottom) {
          candidates.push({ block: b, face: bottom, time: (bottom - fromY) / dy });
        }
      }
    }

    candidates.sort((a, b) => a.time - b.time || (a.block?.id ?? 0) - (b.block?.id ?? 0));
    const hit = candidates[0];
    if (!hit) return;

    if (dy < 0) {
      if (!corrected && this.tryUpwardCornerCorrection(hit.block)) {
        this.resolvePlayerY(fromY, true);
        return;
      }
      p.y = hit.face;
      if (p.yVel > 0) p.yVel = 0;
      return;
    }

    if (!corrected && hit.block?.fixed && this.tryDownwardCornerCorrection(hit.block)) {
      this.resolvePlayerY(fromY, true);
      return;
    }

    p.y = hit.face;
    p.yVel = hit.block?.fixed === false ? -hit.block.yVel : 0;
    p.offGround = 0;
    p.supportBlock = hit.block;
    p.lastWallJumpSide = 0;
    p.clearWallCoyote();
  }

  detectWallContact() {
    const p = this.player;
    const contacts = [];
    if (Math.abs(p.x - PLAYER_MIN_X) <= CONTACT_EPSILON) contacts.push({ side: -1, id: -2 });
    if (Math.abs(p.x + p.w - PLAYER_MAX_X) <= CONTACT_EPSILON) contacts.push({ side: 1, id: -1 });
    for (const b of this.blocks.blocks) {
      if (!overlapsY(p, b)) continue;
      if (Math.abs(p.x + p.w - b.x) <= CONTACT_EPSILON) contacts.push({ side: 1, id: b.id });
      if (Math.abs(p.x - (b.x + b.w)) <= CONTACT_EPSILON) contacts.push({ side: -1, id: b.id });
    }
    if (!contacts.length) return;
    contacts.sort((a, b) => a.id - b.id);
    const preferred = contacts.find(({ side }) => side === p.wallCoyoteSide) ?? contacts[0];
    p.rememberWall(preferred.side);
  }

  wallPass() {
    this.detectWallContact();
  }

  tryUpwardCornerCorrection(block) {
    const p = this.player;
    const candidates = [
      { x: block.x - p.w, direction: -1 },
      { x: block.x + block.w, direction: 1 },
    ]
      .map((candidate) => ({
        ...candidate,
        distance: Math.abs(candidate.x - p.x),
        preference: candidate.direction === Math.sign(p.xVel) ? 0 : 1,
      }))
      .filter((candidate) => candidate.distance <= CORNER_CORRECTION_PX + CONTACT_EPSILON)
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

  tryDownwardCornerCorrection(block) {
    const p = this.player;
    const direction = Math.sign(p.xVel);
    if (direction === 0) return false;
    const x = direction < 0 ? block.x - p.w : block.x + block.w;
    if (Math.abs(x - p.x) > DOWNWARD_CORRECTION_PX + CONTACT_EPSILON) return false;
    if (x < PLAYER_MIN_X || x > PLAYER_MAX_X - p.w) return false;

    const probe = { x, y: p.y, w: p.w, h: p.h };
    const obstructed = this.blocks.blocks.some(
      (other) => other !== block && other.fixed && rectrectStrict(probe, other),
    );
    if (obstructed || rectrectStrict(probe, GROUND)) return false;
    p.x = x;
    return true;
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
      const credited = p.focus < FOCUS_CAP;
      p.highestStableLayer = layer;
      this.events.emit('focusLayer', { gained, layer, credited });
      if (credited) p.focusProgress += gained;
    }
    while (p.focus < FOCUS_CAP && p.focusProgress >= FOCUS_RECHARGE_LAYERS) {
      p.focus++;
      p.focusProgress -= FOCUS_RECHARGE_LAYERS;
      this.events.emit('focusRecharge', { focus: p.focus, layer });
    }
    if (p.focus >= FOCUS_CAP) {
      p.focusProgress = 0;
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
      this.rules.autoGuard ? 1 : 0,
      this.director.phase,
      this.rng.s,
    ].join('|');
  }
}
