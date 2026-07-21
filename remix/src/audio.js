// Synthesized sound effects via Web Audio. Everything is deliberately quiet
// and soft-edged: tones get a short
// linear attack (no clicky onsets) and exponential decay. Frequent sounds
// (block landings) are throttled, pitch-randomized, and volume-ducked so
// late-game block rain never turns into a machine gun.

const MASTER_VOL = 0.35;
const MUTE_KEY = 'dodgeblock-muted';

class Sfx {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.noiseBuf = null;
    this.plays = 0; // debug/testing counter
    this.paused = false;
    this.muted =
      typeof localStorage !== 'undefined' && localStorage.getItem(MUTE_KEY) === '1';
    this.lastBlockLand = 0;
    this.recentLands = [];
  }

  // Must be called from a user-gesture handler (browser autoplay policy).
  // The menu/game-over click handlers do this; safe to call repeatedly.
  init() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') this.ctx.resume();
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = this.muted ? 0 : MASTER_VOL;
    this.master.connect(this.ctx.destination);
    // 0.2s of white noise, reused by every noise burst
    const len = Math.floor(this.ctx.sampleRate * 0.2);
    this.noiseBuf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const data = this.noiseBuf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    if (this.ctx.state === 'suspended') this.ctx.resume();
  }

  toggleMute() {
    this.muted = !this.muted;
    if (this.master) this.master.gain.value = this.muted || this.paused ? 0 : MASTER_VOL;
    try {
      localStorage.setItem(MUTE_KEY, this.muted ? '1' : '0');
    } catch {
      /* private browsing */
    }
    return this.muted;
  }

  setPaused(paused) {
    this.paused = paused;
    if (this.master) this.master.gain.value = this.muted || paused ? 0 : MASTER_VOL;
  }

  get ready() {
    return !!this.ctx && !this.muted && !this.paused;
  }

  tone(freq, end, dur, type = 'sine', vol = 0.15, delay = 0) {
    if (!this.ready) return;
    this.plays++;
    const t0 = this.ctx.currentTime + delay;
    const o = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t0);
    if (end !== freq) o.frequency.exponentialRampToValueAtTime(end, t0 + dur);
    // short attack so the onset never clicks
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.linearRampToValueAtTime(vol, t0 + 0.008);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
    o.connect(g);
    g.connect(this.master);
    o.start(t0);
    o.stop(t0 + dur + 0.02);
  }

  noise(dur, vol, filterFreq, delay = 0) {
    if (!this.ready) return;
    this.plays++;
    const t0 = this.ctx.currentTime + delay;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuf;
    const f = this.ctx.createBiquadFilter();
    f.type = 'lowpass';
    f.frequency.value = filterFreq;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.linearRampToValueAtTime(vol, t0 + 0.005);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
    src.connect(f);
    f.connect(g);
    g.connect(this.master);
    src.start(t0);
    src.stop(t0 + dur + 0.02);
  }

  // --- game sounds ---

  uiClick() {
    this.tone(900, 650, 0.05, 'sine', 0.08);
  }

  jump() {
    this.tone(300, 470, 0.09, 'triangle', 0.12);
  }

  land() {
    this.tone(150, 95, 0.08, 'sine', 0.11);
  }

  // wooden "tok" — throttled and ducked so block rain stays gentle
  blockLand(type = 'wood') {
    if (!this.ready) return;
    const now = performance.now();
    if (now - this.lastBlockLand < 50) return;
    this.lastBlockLand = now;
    this.recentLands = this.recentLands.filter((t) => now - t < 400);
    this.recentLands.push(now);
    const vol = 0.1 / (1 + 0.4 * (this.recentLands.length - 1));
    if (type === 'beam') {
      this.tone(115, 82, 0.13, 'triangle', vol * 1.1);
      this.noise(0.055, vol * 0.45, 1900);
    } else if (type === 'gravel') {
      this.noise(0.065, vol * 0.8, 950);
      this.tone(210, 130, 0.055, 'square', vol * 0.45);
    } else {
      const f = 170 + Math.random() * 60;
      this.tone(f, f * 0.75, 0.07, 'triangle', vol);
      this.noise(0.03, vol * 0.5, 1200);
    }
  }

  branchFault() {
    this.tone(105, 72, 0.42, 'triangle', 0.08);
    this.noise(0.16, 0.035, 420);
  }

  branchShatter() {
    this.tone(145, 72, 0.18, 'triangle', 0.13);
    this.noise(0.13, 0.12, 900);
  }

  focusTick() {
    this.tone(760, 680, 0.045, 'sine', 0.055);
  }

  focusRecharge() {
    this.tone(880, 1320, 0.11, 'triangle', 0.18);
    this.tone(1320, 1760, 0.15, 'sine', 0.15, 0.055);
  }

  autoGuard() {
    this.noise(0.09, 0.14, 1200);
    this.tone(320, 135, 0.14, 'triangle', 0.17);
    this.tone(900, 1250, 0.1, 'sine', 0.11, 0.03);
  }

  // gentle descending "whomp", not a buzzer
  death() {
    this.tone(300, 140, 0.45, 'triangle', 0.15);
    this.tone(150, 70, 0.5, 'sine', 0.13, 0.05);
  }

  dash() {
    this.noise(0.12, 0.1, 2200);
    this.tone(600, 900, 0.12, 'triangle', 0.09);
  }

  focusEnter() {
    this.tone(420, 170, 0.22, 'sine', 0.08);
    this.noise(0.16, 0.045, 520);
  }

  focusRelease() {
    this.tone(180, 720, 0.1, 'triangle', 0.08);
  }

  focusKick() {
    this.tone(95, 62, 0.14, 'sine', 0.16);
    this.noise(0.08, 0.1, 760);
  }

  dashBonk() {
    this.tone(120, 80, 0.08, 'square', 0.11);
    this.noise(0.05, 0.08, 600);
  }

  blockBreak() {
    this.tone(300, 150, 0.08, 'square', 0.08);
    this.noise(0.06, 0.1, 1500);
  }
}

export const sfx = new Sfx();
