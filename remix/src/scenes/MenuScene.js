import Phaser from 'phaser';
import { COLOR_BG_GAME } from '../constants.js';
import { setupCamera, textStyle } from '../utils.js';
import { drawSkyGradient, drawCloud } from '../render/fx.js';
import { drawBlock } from '../render/blockArt.js';
import { drawPlayer } from '../render/playerArt.js';
import { sfx } from '../audio.js';
import { storage } from '../storage.js';
import { normalizeRunRules } from '../rules.js';

const FALLING_BLOCKS = [
  { x: 112, top: 116, bottom: 268, phase: 0.08, shade: 0 },
  { x: 628, top: 104, bottom: 228, phase: 0.5, shade: 2 },
  { x: 688, top: 112, bottom: 268, phase: 0.82, shade: 1 },
];

export class MenuScene extends Phaser.Scene {
  constructor() {
    super('Menu');
  }

  create() {
    setupCamera(this, COLOR_BG_GAME);

    const backdrop = this.add.graphics();
    drawSkyGradient(backdrop, 0x6fb8dd, 0xe2f1ef);
    drawCloud(backdrop, 116, 72, 0.82, 0.56);
    drawCloud(backdrop, 682, 82, 1.05, 0.5);
    drawCloud(backdrop, 442, 142, 0.58, 0.32);

    this.add
      .text(400, 66, 'DODGEBLOCK', textStyle(64, {
        color: '#ffffff',
        fontStyle: 'bold',
        stroke: '#25455d',
        strokeThickness: 9,
      }))
      .setOrigin(0.5)
      .setShadow(0, 5, 'rgba(25, 55, 74, 0.28)', 5);

    this.vignette = this.add.graphics();

    this.rules = normalizeRunRules(storage.data.rules);
    const rulesBand = this.add.graphics();
    rulesBand.fillStyle(0x31566a, 0.14);
    rulesBand.fillRect(0, 188, 800, 112);
    this.add
      .text(400, 202, 'RUN RULES', textStyle(10, {
        color: '#36566c',
        fontStyle: 'bold',
      }))
      .setOrigin(0.5)
      .setAlpha(0.76);
    this.createRuleToggle(228, 'CHECKPOINTS', 'checkpoints');
    this.createRuleToggle(270, 'AUTO GUARD', 'autoGuard');

    const playButton = this.add
      .circle(400, 365, 30, 0xe8433f)
      .setStrokeStyle(2, 0x972d2a)
      .setName('Play')
      .setInteractive({ useHandCursor: true });
    const playIcon = this.add.graphics();
    playIcon.fillStyle(0xffffff, 1);
    playIcon.fillTriangle(393, 351, 393, 379, 414, 365);

    playButton.on('pointerover', () => playButton.setFillStyle(0xf1534f));
    playButton.on('pointerout', () => playButton.setFillStyle(0xe8433f));
    this.tweens.add({
      targets: [playButton, playIcon],
      y: '-=2',
      duration: 900,
      yoyo: true,
      repeat: -1,
      ease: 'Sine.easeInOut',
    });

    this.bestLabel = this.add
      .text(400, 447, '', textStyle(16, {
        color: '#36566c',
        fontStyle: 'bold',
      }))
      .setOrigin(0.5);
    this.refreshBest();

    let starting = false;
    const start = () => {
      if (starting) return;
      starting = true;
      sfx.init();
      sfx.uiClick();
      this.scene.start('Game', {
        rules: this.rules,
      });
    };

    playButton.once('pointerdown', start);
    this.input.keyboard.once('keydown-ENTER', start);
    this.input.keyboard.once('keydown-SPACE', start);
  }

  createRuleToggle(y, label, key) {
    const hit = this.add.container(400, y).setSize(300, 34).setInteractive({ useHandCursor: true });
    const labelText = this.add
      .text(-145, 0, label, textStyle(15, { color: '#31566a', fontStyle: 'bold' }))
      .setOrigin(0, 0.5);
    const track = this.add.graphics();
    const stateText = this.add
      .text(118, 0, '', textStyle(11, { color: '#ffffff', fontStyle: 'bold' }))
      .setOrigin(0.5);
    hit.add([labelText, track, stateText]);

    const draw = () => {
      const enabled = this.rules[key];
      track.clear();
      track.fillStyle(enabled ? 0x318f80 : 0x66747b, 1);
      track.fillRoundedRect(88, -13, 60, 26, 13);
      track.fillStyle(0xffffff, 1);
      track.fillCircle(enabled ? 135 : 101, 0, 10);
      stateText.setText(enabled ? 'ON' : 'OFF').setX(enabled ? 104 : 132);
    };
    hit.on('pointerdown', () => {
      sfx.init();
      sfx.uiClick();
      this.rules = storage.setRules({ ...this.rules, [key]: !this.rules[key] });
      draw();
      this.refreshBest();
    });
    hit.on('pointerover', () => hit.setAlpha(0.82));
    hit.on('pointerout', () => hit.setAlpha(1));
    draw();
  }

  refreshBest() {
    const best = storage.bestForRules(this.rules);
    const mode = this.rules.autoGuard ? 'AUTO GUARD' : 'HARDCORE';
    this.bestLabel
      .setText(`${mode} BEST  ${best.toLocaleString()}`)
      .setAlpha(best > 0 ? 0.9 : 0.58);
  }

  update(time) {
    const gfx = this.vignette;
    gfx.clear();

    // A small playable-looking moment: the character keeps moving and
    // jumping while blocks settle into uneven, climbable terrain.
    for (let i = 0; i < 12; i++) {
      drawBlock(gfx, { x: 40 + i * 60, y: 308, shade: i % 3 });
    }
    drawBlock(gfx, { x: 568, y: 268, shade: 1 });
    drawBlock(gfx, { x: 628, y: 268, shade: 0 });
    drawBlock(gfx, { x: 628, y: 228, shade: 2 });

    for (const block of FALLING_BLOCKS) {
      const progress = (time * 0.00034 + block.phase) % 1;
      const eased = progress * progress;
      drawBlock(gfx, {
        x: block.x,
        y: Phaser.Math.Linear(block.top, block.bottom, eased),
        shade: block.shade,
      });
    }

    const stride = Math.sin(time * 0.00135);
    const jumpPhase = Math.sin(time * 0.0027);
    const jump = Math.max(0, jumpPhase);
    const player = {
      x: 385 + stride * 48,
      y: 278 - Math.pow(jump, 0.72) * 52,
      w: 30,
      h: 30,
      xVel: Math.cos(time * 0.00135) * 2.5,
      yVel: -Math.cos(time * 0.0027) * jump * 5,
      offGround: jump > 0.02 ? 5 : 0,
      landSquash: jump === 0 && jumpPhase > -0.12 ? 5 : 0,
      focus: 2,
      focusTimer: 0,
      focusDX: 0,
      focusDY: 0,
    };
    drawPlayer(gfx, player, Math.floor(time / 16));
  }
}
