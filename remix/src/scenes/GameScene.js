import Phaser from 'phaser';
import {
  BLOCK_H,
  BLOCK_SPAWN_ABOVE,
  BLOCK_W,
  CHECKPOINT_HEIGHT,
  COLOR_BEAM,
  COLOR_BG_GAME,
  COLOR_BLOCK_FILLS,
  COLOR_FOCUS,
  COLOR_GRASS,
  COLOR_GRASS_DARK,
  COLOR_GRAVEL,
  COLOR_SOIL_BOTTOM,
  COLOR_SOIL_TOP,
  COLOR_WARNING,
  COLLAPSE_WARNING_FRAMES,
  FOCUS_AIM_MAX_FRAMES,
  FOCUS_CAP,
  FOCUS_FRAMES,
  FOCUS_RECHARGE_LAYERS,
  FOCUS_SPEED,
  GAME_H,
  GAME_W,
  GROUND,
  MAX_STEPS_PER_FRAME,
  RES,
  STEP_MS,
  TELEGRAPH_FRAMES,
} from '../constants.js';
import { sfx } from '../audio.js';
import { createInput, isMobile } from '../input.js';
import { music } from '../music.js';
import { Background } from '../render/background.js';
import {
  bakeBlockTextures,
  BLOCK_TEX,
  drawWarningStrip,
  frameNameFor,
} from '../render/blockArt.js';
import { stackWarnings, warningUrgency } from '../render/warningLayout.js';
import { ParticleFx } from '../render/fx.js';
import { Juice } from '../render/juice.js';
import { drawPlayer } from '../render/playerArt.js';
import { Sim } from '../sim/sim.js';
import { storage } from '../storage.js';
import { installTestHooks } from '../testhooks.js';
import { normalizeRunRules } from '../rules.js';
import { TouchHints } from '../touchui.js';
import { setupCamera, textStyle } from '../utils.js';
import { ZONES } from '../zones.js';

const params =
  typeof window !== 'undefined'
    ? new URLSearchParams(window.location.search)
    : new Map();

const MATERIAL_COLOR = {
  wood: COLOR_BLOCK_FILLS[0],
  gravel: COLOR_GRAVEL,
  beam: COLOR_BEAM,
};

const WARNING_COLOR = {
  wood: COLOR_WARNING,
  gravel: 0x7b8583,
  beam: COLOR_WARNING,
};
const WARNING_RAIL_Y = 3;
const WARNING_CLEAR_Y = 54;

export class GameScene extends Phaser.Scene {
  constructor() {
    super('Game');
  }

  init(data) {
    this.resumeSnapshot = data?.checkpoint ?? null;
    this.rules = normalizeRunRules(this.resumeSnapshot?.rules ?? data?.rules);
    this.assisted = data?.assisted === true;
    this.continued = data?.continued === true;
    this.replaySeed = Number.isFinite(Number(data?.seed))
      ? Number(data.seed) >>> 0
      : null;
  }

  create() {
    setupCamera(this, COLOR_BG_GAME);
    this.inp = createInput(this);
    const seed = this.resumeSnapshot?.seed ?? (params.has('seed')
      ? Number(params.get('seed')) >>> 0
      : this.replaySeed ?? Date.now() >>> 0);
    this.sim = new Sim(seed, { rules: this.rules });
    if (this.resumeSnapshot) this.sim.restore(this.resumeSnapshot);
    this.checkpointSnapshot = this.rules.checkpoints ? this.resumeSnapshot : null;
    this.checkpointHeight = this.resumeSnapshot
      ? Math.floor(this.sim.height / CHECKPOINT_HEIGHT) * CHECKPOINT_HEIGHT
      : 0;
    this.nextCheckpointHeight = Math.max(
      CHECKPOINT_HEIGHT,
      this.checkpointHeight + CHECKPOINT_HEIGHT,
    );
    this.accumulator = 0;
    this.fx = new ParticleFx();
    this.juice = new Juice();
    this.deathAt = 0;
    this.afterimages = [];
    this.focusPreviewTarget = null;
    this.focusPreviewBranch = new Set();
    this.focusPreviewTopology = -1;
    this.lastFocusCountdownBand = 0;
    this.bestHeightAtStart = storage.bestForRules(this.rules);
    this.wireEvents();

    this.input.keyboard.on('keydown-M', () => sfx.toggleMute());
    this.bg = new Background(this);

    bakeBlockTextures(this);
    this.worldContainer = this.add.container(0, 0);
    this.staticGfx = this.add.graphics();
    this.blockLayer = this.add.container(0, 0);
    this.dynamicGfx = this.add.graphics();
    this.worldContainer.add([this.staticGfx, this.blockLayer, this.dynamicGfx]);
    this.blockSprites = [];

    this.flashGfx = this.add.graphics();
    this.hudGfx = this.add.graphics();
    this.heightLabel = this.add
      .text(18, 15, 'HEIGHT', textStyle(11, {
        color: '#eaf6f6',
        fontStyle: 'bold',
      }))
      .setAlpha(0.78);
    this.heightValue = this.add
      .text(18, 29, '0', textStyle(30, {
        color: '#ffffff',
        fontStyle: 'bold',
        stroke: '#31566a',
        strokeThickness: 4,
      }));
    this.focusLabel = this.add
      .text(671, 25, 'DASH', textStyle(10, {
        color: '#eaf6f6',
        fontStyle: 'bold',
      }))
      .setOrigin(0, 0.5)
      .setAlpha(0.72);
    this.focusProgressLabel = this.add
      .text(765, 43, 'DASH IN 3 NEW LAYERS', textStyle(9, {
        color: '#eaf6f6',
        fontStyle: 'bold',
      }))
      .setOrigin(1, 0.5)
      .setAlpha(0.68);

    if (this.inp.touch && isMobile(this)) new TouchHints(this, this.inp.touch);
    // This stays above the HUD until incoming blocks clear its footprint.
    this.warningGfx = this.add.graphics();
    music.attach(sfx);
    installTestHooks(this);
  }

  wireEvents() {
    const ev = this.sim.events;

    ev.on('blockFix', (b) => {
      if (this.onScreen(b.y)) {
        this.fx.dust(b.x + b.w / 2, b.y + b.h, 5, MATERIAL_COLOR[b.type]);
        sfx.blockLand(b.type);
      }
    });
    ev.on('blockFault', ({ root }) => {
      this.fx.dust(root.x + root.w / 2, root.y + root.h, 4, 0xf2b544);
      sfx.branchFault();
    });
    ev.on('branchShatter', ({ blocks }) => {
      sfx.branchShatter();
      const count = Math.min(18, blocks.length);
      for (let i = 0; i < count; i++) {
        const index = count === 1 ? 0 : Math.round(i * (blocks.length - 1) / (count - 1));
        const block = blocks[index];
        this.fx.shards(
          block.x + block.w / 2,
          block.y + block.h / 2,
          MATERIAL_COLOR[block.type] ?? COLOR_BLOCK_FILLS[0],
          3,
        );
      }
      this.juice.shake(Math.min(6, 2 + blocks.length * 0.25), 180, 2);
    });
    ev.on('jump', () => sfx.jump());
    ev.on('land', ({ x, y }) => {
      this.fx.dust(x, y, 5);
      sfx.land();
    });
    ev.on('focusStart', ({ dx }) => {
      sfx.dash();
      this.juice.leanX = dx * 4;
    });
    ev.on('focusAimStart', () => {
      this.lastFocusCountdownBand = 0;
      sfx.focusEnter();
      music.setFocus(true);
    });
    ev.on('focusAimEnd', ({ guarded } = {}) => {
      if (!guarded) sfx.focusRelease();
      music.setFocus(false);
    });
    ev.on('focusKick', ({ x, y, block }) => {
      sfx.focusKick();
      this.fx.shards(x, y, MATERIAL_COLOR[block.type] ?? COLOR_BLOCK_FILLS[0], 5);
      this.juice.shake(4, 130);
    });
    ev.on('focusEnd', () => {
      this.juice.leanX = 0;
    });
    ev.on('focusBonk', ({ x, y }) => {
      sfx.dashBonk();
      this.juice.shake(2, 90);
      this.fx.dust(x, y, 3);
    });
    ev.on('focusBreak', ({ x, y, block, fixed, affected }) => {
      sfx.blockBreak();
      this.fx.dust(x, y, 7, 0xf2b544);
      this.juice.shake(fixed ? 5 : 3, fixed ? 180 : 100, fixed ? 2 : 1);
      if (affected.length) this.juice.flash(0xf2b544, 0.06, 140);
    });
    ev.on('focusRecharge', () => {
      sfx.focusRecharge();
      const p = this.sim.player;
      this.fx.burst(p.x + p.w / 2, p.y + p.h / 2, COLOR_FOCUS, 7);
      this.tweens.killTweensOf(this.focusLabel);
      this.focusLabel.setScale(1.18).setAlpha(1);
      this.tweens.add({
        targets: this.focusLabel,
        scale: 1,
        alpha: 0.72,
        duration: 260,
        ease: 'Sine.easeOut',
      });
    });
    ev.on('autoGuard', ({ blocks, x, y }) => {
      sfx.autoGuard();
      for (const block of blocks) {
        this.fx.shards(
          block.x + block.w / 2,
          block.y + block.h / 2,
          MATERIAL_COLOR[block.type] ?? COLOR_BLOCK_FILLS[0],
          5,
        );
      }
      this.fx.burst(x + this.sim.player.w / 2, y + this.sim.player.h / 2, COLOR_FOCUS, 10);
      this.juice.shake(5, 180, 3);
      this.juice.flash(COLOR_FOCUS, 0.14, 140);
    });
    ev.on('focusLayer', ({ credited }) => {
      if (!credited) return;
      this.tweens.killTweensOf(this.focusProgressLabel);
      this.focusProgressLabel.setAlpha(1).setScale(1.08);
      this.tweens.add({
        targets: this.focusProgressLabel,
        alpha: 0.68,
        scale: 1,
        duration: 260,
        ease: 'Sine.easeOut',
      });
    });
    ev.on('zoneChange', ({ zone }) => this.bg.setZone(zone));
    ev.on('death', () => {
      sfx.death();
      music.setFocus(false);
      music.duck();
      this.juice.shake(8, 360, 9);
      this.juice.flash(0xffffff, 0.24, 180);
    });
  }

  onScreen(worldY) {
    return worldY + this.sim.camY > -60 && worldY + this.sim.camY < GAME_H + 60;
  }

  update(time, delta) {
    const sim = this.sim;
    this.juice.update(delta);
    this.accumulator += Math.min(delta, 250) * this.juice.timeScale;
    let steps = 0;
    while (this.accumulator >= STEP_MS && steps < MAX_STEPS_PER_FRAME && !sim.dead) {
      sim.step(this.inputSnapshot());
      if (!sim.dead) this.updateCheckpoint();
      if (sim.player.focusAimRemaining > 0 && sim.player.focusAimRemaining <= 30) {
        const band = sim.player.focusAimRemaining > 15 ? 1 : 2;
        if (band !== this.lastFocusCountdownBand) {
          this.lastFocusCountdownBand = band;
          sfx.focusTick();
        }
      }
      this.fx.update();
      this.accumulator -= STEP_MS;
      steps++;
    }
    if (steps === MAX_STEPS_PER_FRAME) this.accumulator = 0;

    music.setPhase(sim.dead ? 'release' : sim.director.phase);

    if (sim.dead) {
      if (!this.deathAt) this.deathAt = time + 550;
      if (time >= this.deathAt) {
        const best = this.assisted
          ? false
          : storage.recordRun({ height: sim.height, seed: sim.seed, rules: this.rules });
        this.scene.start('GameOver', {
          height: sim.height,
          best,
          seed: sim.seed,
          deathCause: sim.deathCause,
          assisted: this.assisted,
          continued: this.continued,
          rules: this.rules,
          checkpoint: this.checkpointSnapshot,
          checkpointHeight: this.checkpointHeight,
        });
        return;
      }
    }

    this.renderWorld();
    this.renderHud();
  }

  updateCheckpoint() {
    if (!this.rules.checkpoints) return;
    const sim = this.sim;
    const p = sim.player;
    if (sim.height < this.nextCheckpointHeight) return;
    if (p.focusAimTimer > 0 || p.focusTimer > 0 || p.offGround !== 0) return;
    if (p.supportBlock && (!p.supportBlock.fixed || p.supportBlock.faultTimer > 0)) return;

    this.checkpointSnapshot = sim.snapshot();
    this.checkpointHeight = Math.floor(sim.height / CHECKPOINT_HEIGHT) * CHECKPOINT_HEIGHT;
    this.nextCheckpointHeight = this.checkpointHeight + CHECKPOINT_HEIGHT;
  }

  inputSnapshot() {
    return {
      up: this.inp.up,
      down: this.inp.down,
      left: this.inp.left,
      right: this.inp.right,
      focusHeld: this.inp.focusHeld,
      ...this.inp.consumePressed(),
    };
  }

  renderWorld() {
    const sim = this.sim;
    const p = sim.player;
    const camY = sim.camY;
    const { ox, oy } = this.juice.offset();
    this.worldContainer.setPosition(ox, camY + oy);
    this.juice.drawFlashes(this.flashGfx, GAME_W, GAME_H);
    this.bg.update(sim.frame, camY, sim.director.pressure ?? 0.5);

    const sg = this.staticGfx;
    sg.clear();
    sg.fillGradientStyle(
      COLOR_SOIL_TOP,
      COLOR_SOIL_TOP,
      COLOR_SOIL_BOTTOM,
      COLOR_SOIL_BOTTOM,
      1,
    );
    sg.fillRect(GROUND.x, GROUND.y, GROUND.w, GROUND.h);
    sg.fillStyle(COLOR_GRASS);
    sg.fillRect(GROUND.x, GROUND.y, GROUND.w, 11);
    sg.fillStyle(COLOR_GRASS_DARK);
    sg.fillRect(GROUND.x, GROUND.y + 11, GROUND.w, 4);

    const dg = this.dynamicGfx;
    dg.clear();
    const warnings = this.warningGfx;
    warnings.clear();
    const warningEntries = [];
    for (const forecast of sim.director.forecasts) {
      const signal = warningUrgency(
        { ...forecast, yVel: 0 },
        forecast.y + camY,
        WARNING_CLEAR_Y,
      );
      warningEntries.push({
        source: forecast,
        x: forecast.x,
        w: forecast.w,
        eta: forecast.frames + signal.eta,
        progress: Math.max(
          1 - forecast.frames / TELEGRAPH_FRAMES,
          signal.progress * 0.75,
        ),
        color: WARNING_COLOR[forecast.type],
      });
    }
    const railTop = -camY - 20;
    const railHeight = GROUND.y - railTop;
    dg.fillStyle(0x31566a, 0.2);
    dg.fillRect(GROUND.x - 3, railTop, 6, railHeight);
    dg.fillRect(GROUND.x + GROUND.w - 3, railTop, 6, railHeight);
    dg.lineStyle(2, 0xffffff, 0.3);
    dg.lineBetween(GROUND.x, railTop, GROUND.x, GROUND.y);
    dg.lineBetween(GROUND.x + GROUND.w, railTop, GROUND.x + GROUND.w, GROUND.y);

    let spriteIdx = 0;
    const blocks = sim.blocks.blocks;
    const stamp = (b) => {
      const img = this.getBlockSprite(spriteIdx++);
      const frame = frameNameFor(b);
      if (img.frame.name !== frame) img.setFrame(frame, false, false);
      img.setPosition(Math.round(b.x), b.y);
      img.setDisplaySize(b.w, b.h);
      const tint = b.type === 'wood' ? (ZONES[b.zone]?.blockTint ?? 0xffffff) : 0xffffff;
      img.setTint(tint).setVisible(true);
    };

    const pulse = (Math.sin(sim.frame * 0.22) + 1) / 2;
    const preview = this.focusInteractionPreview();
    const previewBranch = preview.branch;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (!b.fixed || !this.onScreen(b.y)) continue;
      stamp(b);
      if (previewBranch.has(b) && b.faultTimer <= 0) {
        const isTarget = b === preview.block;
        dg.fillStyle(0xf2b544, (isTarget ? 0.17 : 0.09) + pulse * 0.06);
        dg.fillRect(b.x, b.y, b.w, b.h);
        dg.lineStyle(isTarget ? 3 : 2, 0xf2b544, 0.62 + pulse * 0.18);
        dg.strokeRect(b.x - 1, b.y - 1, b.w + 2, b.h + 2);
      }
      if (b.faultTimer > 0) {
        const progress = 1 - b.faultTimer / (b.faultDuration || COLLAPSE_WARNING_FRAMES);
        const faultColor = 0xf2b544;
        dg.fillStyle(faultColor, 0.09 + pulse * 0.07 + progress * 0.05);
        dg.fillRect(b.x, b.y, b.w, b.h);
        dg.lineStyle(2, faultColor, 0.45 + pulse * 0.25);
        dg.strokeRect(b.x - 1, b.y - 1, b.w + 2, b.h + 2);
        if (b.faultRoot === b) {
          const cx = b.x + b.w / 2;
          const crack = 5 + progress * 11;
          dg.lineStyle(2, faultColor, 0.75 + progress * 0.2);
          dg.lineBetween(cx - 5, b.y + 5, cx + 1, b.y + crack);
          dg.lineBetween(cx + 1, b.y + crack, cx - 3, b.y + crack + 7);
          dg.lineBetween(cx + 1, b.y + crack, cx + 7, b.y + crack + 5);
        }
      }
    }

    for (const b of sim.blocks.falling) {
      const screenY = b.y + camY;
      if (screenY > GAME_H + 80) continue;
      if (screenY < WARNING_CLEAR_Y) {
        const signal = warningUrgency(b, screenY, WARNING_CLEAR_Y, pulse);
        warningEntries.push({
          source: b,
          x: b.x,
          w: b.w,
          eta: signal.eta,
          progress: signal.progress,
          color: WARNING_COLOR[b.type],
        });
      }
      if (b === preview.block && preview.kind === 'falling') {
        dg.fillStyle(COLOR_FOCUS, 0.13 + pulse * 0.07);
        dg.fillRect(b.x, b.y, b.w, b.h);
        dg.lineStyle(3, COLOR_FOCUS, 0.72 + pulse * 0.18);
        dg.strokeRect(b.x - 1, b.y - 1, b.w + 2, b.h + 2);
      }
      if (screenY >= -b.h) stamp(b);
    }
    for (const warning of stackWarnings(warningEntries, { top: WARNING_RAIL_Y })) {
      drawWarningStrip(
        warnings,
        warning.source,
        warning.y,
        warning.progress,
        warning.color,
      );
    }
    for (let i = spriteIdx; i < this.blockSprites.length; i++) {
      this.blockSprites[i].setVisible(false);
    }

    this.drawBestLine(dg);

    if (p.focusTimer > 0) {
      this.afterimages.push({ x: p.x, y: p.y, life: 9 });
      if (this.afterimages.length > 18) this.afterimages.shift();
    }
    for (let i = 0; i < this.afterimages.length; i++) {
      const a = this.afterimages[i];
      dg.fillStyle(COLOR_FOCUS, 0.2 * (a.life / 9));
      dg.fillRoundedRect(a.x, a.y, p.w, p.h, 5);
      if (--a.life <= 0) this.afterimages.splice(i--, 1);
    }

    drawPlayer(dg, p, sim.frame);
    if (p.focusAimTimer > 0) {
      const cx = p.x + p.w / 2;
      const cy = p.y + p.h / 2;
      const length = FOCUS_FRAMES * FOCUS_SPEED;
      const collisionDistance = Number.isFinite(preview.distance)
        ? Math.min(length, preview.distance)
        : length;
      const ex = cx + p.focusDX * collisionDistance;
      const ey = cy + p.focusDY * collisionDistance;
      const nx = -p.focusDY;
      const ny = p.focusDX;
      const remaining = p.focusAimRemaining / FOCUS_AIM_MAX_FRAMES;
      const urgent = p.focusAimRemaining <= 30;
      const baseColor = preview.kind === 'fixed'
        ? 0xf2b544
        : preview.kind === 'bonk'
          ? 0xd9343a
          : COLOR_FOCUS;
      const aimColor = urgent ? 0xf3ffff : baseColor;
      dg.lineStyle(3, aimColor, 0.18);
      dg.strokeCircle(cx, cy, 26);
      dg.lineStyle(3, aimColor, 0.95);
      dg.beginPath();
      dg.arc(cx, cy, 26, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * remaining);
      dg.strokePath();
      dg.lineStyle(3, aimColor, 0.86);
      dg.lineBetween(cx, cy, ex, ey);
      dg.fillStyle(aimColor, 0.92);
      dg.fillTriangle(
        ex,
        ey,
        ex - p.focusDX * 12 + nx * 7,
        ey - p.focusDY * 12 + ny * 7,
        ex - p.focusDX * 12 - nx * 7,
        ey - p.focusDY * 12 - ny * 7,
      );
      if (preview.kind === 'bonk') {
        dg.lineStyle(3, 0xd9343a, 0.95);
        dg.lineBetween(ex - 7, ey - 7, ex + 7, ey + 7);
        dg.lineBetween(ex + 7, ey - 7, ex - 7, ey + 7);
      }
    }
    this.fx.draw(dg);
  }

  focusInteractionPreview() {
    const sim = this.sim;
    const p = sim.player;
    if (p.focusAimTimer <= 0) {
      this.focusPreviewTarget = null;
      this.focusPreviewBranch.clear();
      this.focusPreviewTopology = -1;
      return { kind: 'none', block: null, distance: Infinity, branch: this.focusPreviewBranch };
    }

    const collision = sim.focusPathCollision();
    const target = collision.kind === 'fixed' ? collision.block : null;

    if (!target) {
      this.focusPreviewTarget = null;
      this.focusPreviewBranch.clear();
      this.focusPreviewTopology = -1;
    } else if (
      target !== this.focusPreviewTarget ||
      this.focusPreviewTopology !== sim.blocks.topologyVersion
    ) {
      this.focusPreviewTarget = target;
      this.focusPreviewBranch = new Set(sim.blocks.dependentBranch(target));
      this.focusPreviewTopology = sim.blocks.topologyVersion;
    }
    return { ...collision, branch: this.focusPreviewBranch };
  }

  drawBestLine(gfx) {
    const best = this.bestHeightAtStart;
    if (best <= 0) return;
    const y = GROUND.y - best;
    if (!this.onScreen(y)) return;
    gfx.lineStyle(1.5, 0xffffff, 0.28);
    for (let x = GROUND.x; x < GROUND.x + GROUND.w; x += 28) {
      gfx.lineBetween(x, y, x + 14, y);
    }
  }

  renderHud() {
    const sim = this.sim;
    const p = sim.player;
    const str = String(sim.height);
    if (this.heightValue.text !== str) this.heightValue.setText(str);
    const stored = Math.min(FOCUS_RECHARGE_LAYERS, p.focusProgress);
    const remaining = FOCUS_RECHARGE_LAYERS - stored;
    const noun = remaining === 1 ? 'LAYER' : 'LAYERS';
    const progress = p.focus >= FOCUS_CAP
      ? 'DASHES FULL'
      : remaining === 0
        ? 'DASH READY ON LAND'
        : `DASH IN ${remaining} NEW ${noun}`;
    if (this.focusProgressLabel.text !== progress) {
      this.focusProgressLabel.setText(progress);
    }

    const g = this.hudGfx;
    g.clear();
    for (let i = 0; i < FOCUS_CAP; i++) {
      const cx = 756 - i * 27;
      const cy = 25;
      const filled = i < p.focus;
      g.fillStyle(filled ? COLOR_FOCUS : 0x173b45, filled ? 0.95 : 0.35);
      g.fillPoints([
        new Phaser.Geom.Point(cx, cy - 9),
        new Phaser.Geom.Point(cx + 8, cy),
        new Phaser.Geom.Point(cx, cy + 9),
        new Phaser.Geom.Point(cx - 8, cy),
      ], true);
      g.lineStyle(1.5, 0xffffff, filled ? 0.72 : 0.22);
      g.strokePoints([
        new Phaser.Geom.Point(cx, cy - 9),
        new Phaser.Geom.Point(cx + 8, cy),
        new Phaser.Geom.Point(cx, cy + 9),
        new Phaser.Geom.Point(cx - 8, cy),
      ], true);
    }
    const screenY = p.y + sim.camY;
    if (screenY > 390) {
      const danger = Math.min(1, (screenY - 390) / 90);
      g.fillStyle(0xd9343a, 0.08 + danger * 0.24);
      g.fillRect(0, GAME_H - 8, GAME_W, 8);
    }
  }

  getBlockSprite(index) {
    while (this.blockSprites.length <= index) {
      const image = this.make.image({ key: BLOCK_TEX, frame: 'wood/0', add: false });
      image
        .setOrigin(0, 0)
        .setDisplaySize(BLOCK_W, BLOCK_H)
        .setScale(1 / RES)
        .setVisible(false);
      this.blockLayer.add(image);
      this.blockSprites.push(image);
    }
    return this.blockSprites[index];
  }
}
