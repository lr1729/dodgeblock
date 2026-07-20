// True on phones/tablets only: touch-capable AND the primary pointer is
// coarse. Touchscreen laptops fail the second check (their primary pointer
// is the mouse/trackpad), so they keep the normal desktop presentation —
// the touch layer still works there, it's just not advertised.
export function isMobile(scene) {
  return (
    scene.sys.game.device.input.touch &&
    typeof window !== 'undefined' &&
    window.matchMedia('(pointer: coarse)').matches
  );
}

// Held-key state for arrows + WASD (mirroring the original's keys[] array)
// plus edge-detected presses. Presses accumulate between sim steps and are
// consumed exactly once per step via consumePressed() — a tap that lands
// between two steps is never lost, even under hitstop.
//
// Focus Aim: hold Shift / X / K plus a direction, then release to commit.
export function createInput(scene) {
  const k = scene.input.keyboard.addKeys('UP,DOWN,LEFT,RIGHT,W,A,S,D,SPACE');
  const pressed = {
    jump: false,
    focus: false,
    focusReleased: false,
    focusDirX: 0,
    focusDirY: 0,
  };
  const focusKeys = new Set();

  scene.input.keyboard.on('keydown', (e) => {
    if (e.repeat) return;
    switch (e.code) {
      case 'ArrowUp':
      case 'KeyW':
      case 'Space':
        pressed.jump = true;
        break;
      case 'ShiftLeft':
      case 'ShiftRight':
      case 'KeyX':
      case 'KeyK':
        if (!focusKeys.size) pressed.focus = true;
        focusKeys.add(e.code);
        break;
    }
  });
  scene.input.keyboard.on('keyup', (e) => {
    if (!focusKeys.delete(e.code)) return;
    if (!focusKeys.size) pressed.focusReleased = true;
  });
  scene.input.keyboard.addCapture('SPACE,SHIFT,X,K,UP,DOWN,LEFT,RIGHT,W,A,S,D');

  const touch = scene.sys.game.device.input.touch
    ? createTouch(scene, pressed)
    : null;

  return {
    touch, // null on non-touch devices; used by TouchHints
    get up() {
      return k.UP.isDown || k.W.isDown || k.SPACE.isDown || !!(touch && touch.jump);
    },
    get down() {
      return k.DOWN.isDown || k.S.isDown;
    },
    get left() {
      return k.LEFT.isDown || k.A.isDown || !!(touch && touch.dir(-1));
    },
    get right() {
      return k.RIGHT.isDown || k.D.isDown || !!(touch && touch.dir(1));
    },
    get focusHeld() {
      return focusKeys.size > 0 || !!(touch && touch.focusHeld);
    },
    // fresh presses since the last call; cleared on read
    consumePressed() {
      const out = {
        jumpPressed: pressed.jump,
        focusPressed: pressed.focus,
        focusReleased: pressed.focusReleased,
        focusDirX: pressed.focusDirX || touch?.focusDirX || 0,
        focusDirY: pressed.focusDirY || touch?.focusDirY || 0,
      };
      pressed.jump = pressed.focus = pressed.focusReleased = false;
      pressed.focusDirX = pressed.focusDirY = 0;
      return out;
    },
  };
}

// Touch zones (fractions of the canvas, so RES scaling is irrelevant):
// top 40% = jump, bottom 60% split at the middle = left / right.
// A pointer's role is locked to the zone it STARTED in — a movement thumb
// that drifts upward never triggers an accidental jump; only a fresh tap in
// the jump zone does. Movement pointers re-read their current x each frame,
// so sliding a held thumb across the middle switches direction.
export const JUMP_ZONE_FRAC = 0.4;

// Focus is a deliberate hold-then-swipe. The lower bound prevents a quick
// movement-thumb reversal from spending a charge; the upper bound keeps an
// old movement touch from unexpectedly becoming Focus much later.
const SWIPE_MIN_MS = 120;
const SWIPE_MAX_MS = 700;
const SWIPE_FRAC = 0.05;

function createTouch(scene, pressed) {
  const active = new Map(); // pointer.id -> { role, pointer, x0, y0, t0, swiped }
  const halfW = () => scene.scale.width / 2;

  const touch = {
    onZonePress: null, // TouchHints hook: (zone: 'jump'|'left'|'right') => void
    onFocus: null,
    get jump() {
      for (const { role, pointer } of active.values()) {
        if (role === 'jump' && pointer.isDown) return true;
      }
      return false;
    },
    get focusDirX() {
      for (const e of active.values()) if (e.swiped) return e.focusDirX;
      return 0;
    },
    get focusDirY() {
      for (const e of active.values()) if (e.swiped) return e.focusDirY;
      return 0;
    },
    dir(sign) {
      for (const { role, pointer } of active.values()) {
        if (role !== 'move' || !pointer.isDown) continue;
        if (sign < 0 ? pointer.x < halfW() : pointer.x >= halfW()) return true;
      }
      return false;
    },
    get focusHeld() {
      for (const e of active.values()) {
        if (e.swiped && e.pointer.isDown) return true;
      }
      return false;
    },
  };

  const begin = (p) => {
    const zone =
      p.y < scene.scale.height * JUMP_ZONE_FRAC
        ? 'jump'
        : p.x < halfW()
          ? 'left'
          : 'right';
    active.set(p.id, {
      role: zone === 'jump' ? 'jump' : 'move',
      pointer: p,
      x0: p.x,
      y0: p.y,
      t0: p.downTime ?? p.timeStamp ?? performance.now(),
      swiped: false,
      disqualified: false,
      focusDirX: 0,
      focusDirY: 0,
    });
    if (zone === 'jump') pressed.jump = true;
    if (touch.onZonePress) touch.onZonePress(zone);
  };

  const move = (p) => {
    const e = active.get(p.id);
    if (!e || !p.isDown) return;
    if (e.role !== 'move') return;
    const dx = p.x - e.x0;
    const dy = p.y - e.y0;
    const thresh = scene.scale.width * SWIPE_FRAC;
    const elapsed = (p.timeStamp ?? performance.now()) - e.t0;
    const quickVertical = Math.abs(dy) >= thresh && Math.abs(dy) > Math.abs(dx) * 1.2;
    if (
      elapsed < SWIPE_MIN_MS &&
      !quickVertical &&
      Math.max(Math.abs(dx), Math.abs(dy)) >= thresh * 0.5
    ) {
      e.disqualified = true;
    }
    if (
      !e.swiped &&
      !e.disqualified &&
      (elapsed >= SWIPE_MIN_MS || quickVertical) &&
      elapsed <= SWIPE_MAX_MS &&
      Math.max(Math.abs(dx), Math.abs(dy)) >= thresh
    ) {
      e.swiped = true;
      pressed.focus = true;
      touch.onFocus?.();
    }
    if (e.swiped) {
      e.focusDirX = Math.abs(dx) >= thresh * 0.55 ? Math.sign(dx) : 0;
      e.focusDirY = Math.abs(dy) >= thresh * 0.55 ? Math.sign(dy) : 0;
      pressed.focusDirX = e.focusDirX;
      pressed.focusDirY = e.focusDirY;
    }
  };

  const drop = (p) => {
    const e = active.get(p.id);
    if (e?.swiped) pressed.focusReleased = true;
    active.delete(p.id);
  };
  scene.input.on('pointerdown', begin);
  scene.input.on('pointermove', move);
  scene.input.on('pointerup', drop);
  scene.input.on('pointerupoutside', drop);

  // Phaser receives events only on the letterboxed canvas. Capture touches
  // that begin in the side margins and map them to the nearest logical edge.
  if (typeof window !== 'undefined') {
    const canvas = scene.sys.game.canvas;
    const mapped = new Map();
    const mapEvent = (event, isDown = true) => {
      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
      return {
        id: `margin-${event.pointerId}`,
        x: x * scene.scale.width,
        y: y * scene.scale.height,
        downTime: event.timeStamp,
        timeStamp: event.timeStamp,
        isDown,
      };
    };
    const outsideCanvas = (event) => event.target !== canvas && !canvas.contains(event.target);
    const onDown = (event) => {
      if (event.pointerType !== 'touch' || !outsideCanvas(event)) return;
      const pointer = mapEvent(event);
      mapped.set(event.pointerId, pointer);
      begin(pointer);
      event.preventDefault();
    };
    const onMove = (event) => {
      const pointer = mapped.get(event.pointerId);
      if (!pointer) return;
      Object.assign(pointer, mapEvent(event));
      move(pointer);
      event.preventDefault();
    };
    const onUp = (event) => {
      const pointer = mapped.get(event.pointerId);
      if (!pointer) return;
      Object.assign(pointer, mapEvent(event, false));
      drop(pointer);
      mapped.delete(event.pointerId);
      event.preventDefault();
    };
    window.addEventListener('pointerdown', onDown, { passive: false });
    window.addEventListener('pointermove', onMove, { passive: false });
    window.addEventListener('pointerup', onUp, { passive: false });
    window.addEventListener('pointercancel', onUp, { passive: false });
    scene.events.once('shutdown', () => {
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    });
  }
  return touch;
}
