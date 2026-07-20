// Test bridge, active only with ?test in the URL. Exposes the live sim and a
// synchronous fast-forward so Playwright can drive thousands of frames
// without waiting on requestAnimationFrame.

import { NEUTRAL_INPUT } from './sim/sim.js';
import { music } from './music.js';

export function installTestHooks(scene) {
  if (typeof window === 'undefined') return;
  if (!new URLSearchParams(window.location.search).has('test')) return;

  const sim = scene.sim;
  const log = [];
  sim.events.tap = (name, payload) => {
    log.push({ frame: sim.frame, name, payload });
    if (log.length > 20000) log.splice(0, 10000);
  };

  window.__scene = scene;
  window.__sim = sim;
  window.__juice = scene.juice;
  window.__music = music;
  window.__events = log;
  window.__snap = () => ({
    frame: sim.frame,
    x: sim.player.x,
    y: sim.player.y,
    blocks: sim.blocks.length,
    height: sim.height,
    focus: sim.player.focus,
    focusAimTimer: sim.player.focusAimTimer,
    focusAimRemaining: sim.player.focusAimRemaining,
    focusTimer: sim.player.focusTimer,
    focusProgress: sim.player.focusProgress,
    rules: sim.rules,
    phase: sim.director.phase,
    stormFrames: sim.director.elapsedFrames,
    forecasts: sim.director.forecasts.length,
    faults: sim.blocks.faults.length,
    camY: sim.camY,
    hash: sim.hash(),
  });
  window.__ff = (n, input = NEUTRAL_INPUT) => {
    for (let i = 0; i < n && !sim.dead; i++) sim.step(input);
    return window.__snap();
  };
}
