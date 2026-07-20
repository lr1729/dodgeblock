import {
  COYOTE_FRAMES,
  GRAVITY,
  JUMP_VEL,
  MOVE_ACCEL_AIR,
  MOVE_ACCEL_GROUND,
  MOVE_FRICTION,
  MOVE_SPEED,
  FOCUS_CAP,
  FOCUS_RECHARGE_LAYERS,
  PLAYER_FALL_CAP,
  PLAYER_MAX_X,
  PLAYER_MIN_X,
  PLAYER_SIZE,
  PLAYER_START_X,
  PLAYER_START_Y,
  WALL_JUMP_X,
  WALL_JUMP_Y,
  WALL_COYOTE_FRAMES,
} from '../constants.js';
import { constrain } from './util.js';

function approach(value, target, amount) {
  if (value < target) return Math.min(value + amount, target);
  if (value > target) return Math.max(value - amount, target);
  return value;
}

export class Player {
  constructor() {
    this.x = PLAYER_START_X;
    this.y = PLAYER_START_Y;
    this.w = PLAYER_SIZE;
    this.h = PLAYER_SIZE;
    this.xVel = 0;
    this.yVel = 0;
    this.facing = 1;

    this.offGround = 10;
    this.timeSinceJump = 10;
    this.originalPos = this.x;
    this.landSquash = 0;
    this.supportBlock = null;
    this.wallSide = 0;
    this.wallCoyoteFrames = 0;
    this.wallCoyoteSide = 0;
    this.lastWallJumpSide = 0;

    this.focus = FOCUS_CAP;
    this.focusAimTimer = 0;
    this.focusAimRemaining = 0;
    this.focusTimer = 0;
    this.focusDX = 0;
    this.focusDY = 0;
    this.stableFrames = 0;
    this.highestStableLayer = 0;
    this.focusProgress = 0;
    this.nextFocusLayer = FOCUS_RECHARGE_LAYERS;
  }

  move(axis, timeScale = 1) {
    if (axis !== 0) {
      const accel = this.offGround <= COYOTE_FRAMES ? MOVE_ACCEL_GROUND : MOVE_ACCEL_AIR;
      this.xVel = approach(this.xVel, axis * MOVE_SPEED, accel * timeScale);
      this.facing = axis;
    } else {
      this.xVel *= Math.pow(MOVE_FRICTION, timeScale);
      if (Math.abs(this.xVel) < 0.05) this.xVel = 0;
    }
  }

  jump() {
    if (this.timeSinceJump <= 2) return false;
    const grounded = this.offGround <= COYOTE_FRAMES;
    const rememberedWall = this.wallSide || this.wallCoyoteSide;
    const canWallJump =
      !grounded &&
      rememberedWall !== 0 &&
      rememberedWall !== this.lastWallJumpSide;
    if (!grounded && !canWallJump) return false;

    if (canWallJump) {
      this.yVel = WALL_JUMP_Y;
      this.xVel = -rememberedWall * WALL_JUMP_X;
      this.lastWallJumpSide = rememberedWall;
      this.wallCoyoteFrames = 0;
      this.wallCoyoteSide = 0;
    } else {
      this.yVel = JUMP_VEL;
      this.lastWallJumpSide = 0;
    }
    this.timeSinceJump = 0;
    this.offGround = COYOTE_FRAMES + 1;
    this.supportBlock = null;
    this.wallSide = 0;
    return true;
  }

  rememberWall(side) {
    this.wallSide = side;
    this.wallCoyoteSide = side;
    this.wallCoyoteFrames = WALL_COYOTE_FRAMES;
  }

  updateWallCoyote(timeScale = 1) {
    if (this.wallSide !== 0) return;
    this.wallCoyoteFrames = Math.max(0, this.wallCoyoteFrames - timeScale);
    if (this.wallCoyoteFrames <= 0) this.wallCoyoteSide = 0;
  }

  clearWallCoyote() {
    this.wallSide = 0;
    this.wallCoyoteFrames = 0;
    this.wallCoyoteSide = 0;
  }

  updateX(timeScale = 1) {
    this.originalPos = this.x;
    this.x += this.xVel * timeScale;
    this.clampX();
  }

  clampX() {
    const bounded = constrain(this.x, PLAYER_MIN_X, PLAYER_MAX_X - this.w);
    if (bounded !== this.x) {
      this.wallSide = this.x < PLAYER_MIN_X ? -1 : 1;
      this.xVel = 0;
    }
    this.x = bounded;
  }

  updateY(upHeld, downHeld, timeScale = 1) {
    const rising = this.yVel > 0;
    const gravityMul = rising && !upHeld ? 1.75 : 1;
    this.yVel -= GRAVITY * gravityMul * timeScale;
    if (downHeld && this.yVel < 0) this.yVel -= GRAVITY * 0.65 * timeScale;
    this.yVel = Math.max(this.yVel, -PLAYER_FALL_CAP);
    this.y -= this.yVel * timeScale;
  }
}
