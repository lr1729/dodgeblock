import { GAME_W, GAME_H } from './constants.js';
import { textStyle } from './utils.js';
import { JUMP_ZONE_FRAC } from './input.js';

// Render-only touch-control hints: whisper-faint persistent zone dividers,
// per-zone icons that fade once the zone has been used a few times, and a
// brief zone flash on every touch as passive feedback.

const JUMP_H = GAME_H * JUMP_ZONE_FRAC;
const ZONES = {
  jump: { x: 0, y: 0, w: GAME_W, h: JUMP_H },
  left: { x: 0, y: JUMP_H, w: GAME_W / 2, h: GAME_H - JUMP_H },
  right: { x: GAME_W / 2, y: JUMP_H, w: GAME_W / 2, h: GAME_H - JUMP_H },
};

const LEARNED_AT = 3; // presses of a zone before its icon fades
const ICON_ALPHA = 0.16;
const ICON_ALPHA_LEARNED = 0.04;
const FLASH_ALPHA = 0.09;

// survives scene restarts within a session, resets on page reload
const uses = { jump: 0, left: 0, right: 0 };
let focusLearned = false;

export class TouchHints {
  constructor(scene, touch) {
    this.scene = scene;

    const lines = scene.add.graphics().setAlpha(0.1);
    lines.lineStyle(2, 0xffffff, 1);
    lines.lineBetween(0, JUMP_H, GAME_W, JUMP_H);
    lines.lineBetween(GAME_W / 2, JUMP_H, GAME_W / 2, GAME_H);

    this.flashGfx = scene.add.graphics().setAlpha(0);

    const icon = (str, x, y, zone) =>
      scene.add
        .text(x, y, str, textStyle(44, { color: '#ffffff', fontStyle: 'bold' }))
        .setOrigin(0.5)
        .setAlpha(uses[zone] >= LEARNED_AT ? ICON_ALPHA_LEARNED : ICON_ALPHA);
    this.icons = {
      jump: icon('▲', GAME_W / 2, JUMP_H / 2, 'jump'),
      left: icon('◀', GAME_W / 4, JUMP_H + (GAME_H - JUMP_H) / 2, 'left'),
      right: icon('▶', (GAME_W * 3) / 4, JUMP_H + (GAME_H - JUMP_H) / 2, 'right'),
    };

    this.focusHint = scene.add.graphics().setAlpha(focusLearned ? 0 : 0.2);
    this.focusHint.lineStyle(3, 0x37d6d0, 1);
    this.focusHint.strokeCircle(690, 416, 7);
    this.focusHint.lineBetween(690, 405, 690, 350);
    this.focusHint.fillStyle(0x37d6d0, 1);
    this.focusHint.fillTriangle(682, 359, 698, 359, 690, 347);
    if (!focusLearned) {
      scene.tweens.add({
        targets: this.focusHint,
        y: -6,
        alpha: { from: 0.1, to: 0.22 },
        duration: 850,
        yoyo: true,
        repeat: -1,
        ease: 'Sine.easeInOut',
      });
    }

    touch.onZonePress = (zone) => this.press(zone);
    touch.onFocus = () => {
      if (focusLearned) return;
      focusLearned = true;
      scene.tweens.killTweensOf(this.focusHint);
      scene.tweens.add({ targets: this.focusHint, alpha: 0, duration: 350 });
    };
  }

  press(zone) {
    const z = ZONES[zone];
    const g = this.flashGfx;
    g.clear();
    g.fillStyle(0xffffff, 1);
    g.fillRect(z.x, z.y, z.w, z.h);
    this.scene.tweens.killTweensOf(g);
    g.setAlpha(FLASH_ALPHA);
    this.scene.tweens.add({ targets: g, alpha: 0, duration: 220 });

    if (++uses[zone] === LEARNED_AT) {
      this.scene.tweens.add({
        targets: this.icons[zone],
        alpha: ICON_ALPHA_LEARNED,
        duration: 800,
      });
    }
  }
}
