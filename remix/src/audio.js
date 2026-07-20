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
    if (this.master) this.master.gain.value = this.muted ? 0 : MASTER_VOL;
    try {
      localStorage.setItem(MUTE_KEY, this.muted ? '1' : '0');
    } catch {
      /* private browsing */
    }
    return this.muted;
  }

  get ready() {
    return !!this.ctx && !this.muted;
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

  // warm two-note chime for shield/speed/jump/boost
  pickup() {
    this.tone(523, 523, 0.1, 'sine', 0.14);
    this.tone(784, 784, 0.16, 'sine', 0.13, 0.07);
  }

  // brighter double ding for coins
  coin() {
    this.tone(988, 988, 0.08, 'sine', 0.12);
    this.tone(1319, 1319, 0.14, 'sine', 0.11, 0.06);
  }

  // shield eating a block: soft pop
  shieldPop() {
    this.noise(0.06, 0.12, 800);
    this.tone(280, 140, 0.12, 'sine', 0.12);
  }

  // gentle descending "whomp", not a buzzer
  death() {
    this.tone(300, 140, 0.45, 'triangle', 0.15);
    this.tone(150, 70, 0.5, 'sine', 0.13, 0.05);
  }

  // --- remix sounds ---
  // mix discipline: sub-100Hz is reserved for the three big impacts (spike,
  // crest, death); grazes/phases live in high noise; melodic 400-1300Hz.

  // "shhk" — air ripping past
  graze(perfect) {
    this.noise(0.06, 0.09, 3200);
    if (perfect) this.tone(1200, 1400, 0.09, 'sine', 0.11);
  }

  // airy whoosh
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

  // chunky "zzip" per phased block
  dashPhase() {
    this.tone(220, 260, 0.05, 'square', 0.07);
  }

  spikeWindup() {
    this.tone(500, 740, 0.09, 'sine', 0.07);
  }

  // the lowest, heaviest sound in the palette
  spikeShatter() {
    this.tone(50, 50, 0.28, 'sine', 0.22);
    this.tone(200, 90, 0.16, 'square', 0.09);
    this.noise(0.12, 0.16, 900);
  }

  spikeThud() {
    this.tone(90, 60, 0.1, 'sine', 0.14);
    this.noise(0.05, 0.08, 500);
  }

  // mid-air block shatter during a spike drop
  blockBreak() {
    this.tone(300, 150, 0.08, 'square', 0.08);
    this.noise(0.06, 0.1, 1500);
  }

  // ridden block slams home — deep thump
  crestThump() {
    this.tone(60, 40, 0.3, 'sine', 0.2);
    this.noise(0.1, 0.12, 400);
  }

  // stacked-fifth chime for the super jump
  crestJump() {
    this.tone(440, 440, 0.2, 'triangle', 0.12);
    this.tone(660, 660, 0.26, 'triangle', 0.11, 0.03);
  }

  // pentatonic ladder — every Heat gain plays the next step up (the Peggle
  // trick); resets as Heat decays because the step IS the heat value
  ladder(step) {
    const NOTES = [523, 587, 659, 784, 880, 1047, 1175, 1319];
    const f = NOTES[Math.max(0, Math.min(step - 1, NOTES.length - 1))];
    this.tone(f, f, 0.14, 'triangle', 0.09);
  }

  heatDown() {
    this.tone(400, 300, 0.12, 'triangle', 0.06);
  }

  // zone fanfare: rising major triad
  zoneUp() {
    this.tone(523, 523, 0.16, 'triangle', 0.12);
    this.tone(659, 659, 0.16, 'triangle', 0.12, 0.09);
    this.tone(784, 784, 0.28, 'triangle', 0.12, 0.18);
  }

  // --- event telegraphs ---

  siren() {
    this.tone(520, 780, 0.5, 'sine', 0.09);
    this.tone(780, 520, 0.5, 'sine', 0.09, 0.5);
    this.tone(520, 780, 0.5, 'sine', 0.09, 1.0);
  }

  rumble() {
    this.tone(45, 38, 1.6, 'sine', 0.2);
    this.noise(1.2, 0.08, 220);
  }

  whistle() {
    this.tone(600, 1900, 1.6, 'sine', 0.07);
  }

  gustWhoosh() {
    this.noise(0.9, 0.1, 1400);
  }

  shimmer() {
    for (let i = 0; i < 4; i++) {
      const f = [1047, 1319, 1568, 2093][i];
      this.tone(f, f, 0.18, 'sine', 0.06, i * 0.12);
    }
  }

  // block spawning inside a whiteout — the audio you dodge by
  spawnTick() {
    this.tone(1100, 950, 0.04, 'square', 0.06);
  }

  payoffChime() {
    this.tone(880, 880, 0.12, 'triangle', 0.12);
    this.tone(1109, 1109, 0.14, 'triangle', 0.11, 0.08);
    this.tone(1319, 1319, 0.2, 'triangle', 0.11, 0.16);
  }

  meteorBoom() {
    this.tone(55, 40, 0.4, 'sine', 0.22);
    this.noise(0.25, 0.18, 700);
  }
}

export const sfx = new Sfx();
