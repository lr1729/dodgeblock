import Phaser from 'phaser';
import {
  COLOR_DEAD_SKY_BOTTOM,
  COLOR_DEAD_SKY_TOP,
  COLOR_PLAYER,
} from '../constants.js';
import { setupCamera, textStyle } from '../utils.js';
import { drawSkyGradient, drawCloud } from '../render/fx.js';
import { sfx } from '../audio.js';
import { storage } from '../storage.js';

const DEATH_CAUSES = {
  squished: 'Crushed from above',
  fell: 'Lost below the climb',
};

export class GameOverScene extends Phaser.Scene {
  constructor() {
    super('GameOver');
  }

  init(data) {
    this.result = data ?? {};
  }

  create() {
    setupCamera(this, COLOR_DEAD_SKY_TOP);

    const g = this.add.graphics();
    drawSkyGradient(g, COLOR_DEAD_SKY_TOP, COLOR_DEAD_SKY_BOTTOM);
    drawCloud(g, 118, 112, 0.95, 0.13);
    drawCloud(g, 686, 84, 0.78, 0.12);

    const runHeight = Number(this.result.height);
    const height = Number.isFinite(runHeight) ? Math.max(0, Math.round(runHeight)) : 0;
    const newBest = this.result.best === true;
    const reportedBest = typeof this.result.best === 'number'
      ? this.result.best
      : storage.data.bestHeight;
    const numericBest = Number(reportedBest);
    const storedBest = Number.isFinite(numericBest) ? Math.round(numericBest) : 0;
    const best = this.result.assisted ? storedBest : Math.max(height, storedBest);

    this.add
      .text(400, 74, 'GAME OVER', textStyle(30, {
        color: '#edf3f5',
        fontStyle: 'bold',
      }))
      .setOrigin(0.5)
      .setAlpha(0.88);

    this.add
      .text(400, 132, 'HEIGHT', textStyle(14, {
        color: '#d5dfe4',
        fontStyle: 'bold',
      }))
      .setOrigin(0.5)
      .setAlpha(0.72);

    this.add
      .text(400, 202, height.toLocaleString(), textStyle(76, {
        color: Phaser.Display.Color.IntegerToColor(COLOR_PLAYER).rgba,
        fontStyle: 'bold',
        stroke: '#f4f6f7',
        strokeThickness: 6,
      }))
      .setOrigin(0.5);

    this.add
      .text(400, 258, `${newBest ? 'NEW BEST' : 'BEST'}  ${best.toLocaleString()}`, textStyle(17, {
        color: '#edf3f5',
        fontStyle: 'bold',
      }))
      .setOrigin(0.5)
      .setAlpha(0.78);

    const cause = DEATH_CAUSES[this.result.deathCause];
    if (cause) {
      this.add
        .text(400, 296, cause, textStyle(15, { color: '#d5dfe4' }))
        .setOrigin(0.5)
        .setAlpha(0.66);
    }

    if (this.result.assisted) {
      this.add
        .text(400, 326, 'CHECKPOINT RUN', textStyle(11, {
          color: '#d5dfe4',
          fontStyle: 'bold',
        }))
        .setOrigin(0.5)
        .setAlpha(0.52);
    }

    const checkpoint = this.result.checkpoint
      ? this.command(250, 374, 'CHECKPOINT', true)
      : null;
    const replay = this.command(checkpoint ? 410 : 330, 374, 'REPLAY', !checkpoint);
    const menu = this.command(checkpoint ? 550 : 470, 374, 'MENU', false);

    const seed = this.result.seed;
    if (seed !== undefined && seed !== null && seed !== '') {
      this.add
        .text(400, 447, `seed ${seed}`, textStyle(10, { color: '#d5dfe4' }))
        .setOrigin(0.5)
        .setAlpha(0.38);
    }

    let leaving = false;
    const go = (scene, data) => {
      if (leaving) return;
      leaving = true;
      sfx.init();
      sfx.uiClick();
      this.scene.start(scene, data);
    };
    const restart = () => go('Game', { seed });
    const resume = () => go('Game', {
      seed,
      checkpoint: this.result.checkpoint,
      assisted: true,
    });
    const toMenu = () => go('Menu');

    // Ignore the input that may still be held when the death transition lands.
    this.time.delayedCall(350, () => {
      replay.setInteractive({ useHandCursor: true });
      menu.setInteractive({ useHandCursor: true });
      checkpoint?.setInteractive({ useHandCursor: true });
      checkpoint?.once('pointerdown', resume);
      replay.once('pointerdown', restart);
      menu.once('pointerdown', toMenu);
      this.input.keyboard.once('keydown-R', restart);
      this.input.keyboard.once('keydown-C', checkpoint ? resume : restart);
      this.input.keyboard.once('keydown-ENTER', checkpoint ? resume : restart);
      this.input.keyboard.once('keydown-SPACE', checkpoint ? resume : restart);
      this.input.keyboard.once('keydown-ESC', toMenu);
      this.input.keyboard.once('keydown-M', toMenu);
    });
  }

  command(x, y, label, primary) {
    const command = this.add
      .text(x, y, label, textStyle(20, {
        color: primary ? '#ffffff' : '#e1e8eb',
        backgroundColor: primary ? '#e8433f' : '#46535d',
        fontStyle: 'bold',
        padding: { x: 22, y: 12 },
      }))
      .setOrigin(0.5);

    command.on('pointerover', () => command.setAlpha(0.82));
    command.on('pointerout', () => command.setAlpha(1));
    return command;
  }
}
