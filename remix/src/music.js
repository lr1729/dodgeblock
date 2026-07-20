// Sparse procedural pressure bed. Calm and Release are intentionally silent;
// Build adds a low pulse and Surge adds an irregular high-noise layer. There is
// no melody loop or removed Heat state to climb forever in the background.

const BPM = 96;
const STEP_DUR = 60 / BPM / 2;
const STEPS = 16;

class Music {
  constructor() {
    this.sfx = null;
    this.bus = null;
    this.filter = null;
    this.intensity = 0;
    this.phase = 'opening';
    this.timer = null;
    this.step = 0;
    this.nextNoteTime = 0;
    this.scheduledNotes = 0;
  }

  attach(sfx) {
    if (this.bus || !sfx.ctx) return;
    this.sfx = sfx;
    const ctx = sfx.ctx;
    this.bus = ctx.createGain();
    this.bus.gain.value = 0.42;
    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.value = 16000;
    this.bus.connect(this.filter);
    this.filter.connect(sfx.master);
    this.nextNoteTime = ctx.currentTime + 0.1;
    this.timer = setInterval(() => this.schedule(), 25);
  }

  setPhase(phase) {
    this.phase = phase;
    this.intensity = phase === 'surge' ? 2 : phase === 'build' ? 1 : 0;
  }

  // Compatibility for test/dev poking; gameplay uses setPhase.
  setIntensity(value) {
    this.intensity = Math.max(0, Math.min(2, value));
  }

  setFocus(active) {
    if (!this.filter || !this.sfx?.ctx) return;
    const t = this.sfx.ctx.currentTime;
    const f = this.filter.frequency;
    f.cancelScheduledValues(t);
    f.setValueAtTime(Math.max(300, f.value), t);
    f.exponentialRampToValueAtTime(active ? 620 : 16000, t + (active ? 0.08 : 0.14));
    const gain = this.bus.gain;
    gain.cancelScheduledValues(t);
    gain.setValueAtTime(gain.value, t);
    gain.linearRampToValueAtTime(active ? 0.24 : 0.42, t + 0.1);
  }

  duck() {
    if (!this.bus || !this.sfx?.ctx) return;
    const t = this.sfx.ctx.currentTime;
    const gain = this.bus.gain;
    gain.cancelScheduledValues(t);
    gain.setValueAtTime(0.1, t);
    gain.linearRampToValueAtTime(0.42, t + 0.35);
  }

  schedule() {
    const ctx = this.sfx?.ctx;
    if (!ctx || ctx.state !== 'running') return;
    while (this.nextNoteTime < ctx.currentTime + 0.12) {
      if (this.intensity > 0 && !this.sfx.muted) this.playStep(this.step, this.nextNoteTime);
      this.step = (this.step + 1) % STEPS;
      this.nextNoteTime += STEP_DUR;
    }
  }

  playStep(step, time) {
    if (step === 0 || step === 8 || (this.intensity >= 2 && (step === 4 || step === 12))) {
      const freq = step < 8 ? 82.41 : 73.42;
      this.pulse(freq, time, this.intensity >= 2 ? 0.1 : 0.065);
    }
    if (this.intensity >= 2 && [2, 6, 11, 14].includes(step)) {
      this.airTick(time, step === 11 ? 0.035 : 0.025);
    }
  }

  pulse(freq, time, volume) {
    this.scheduledNotes++;
    const ctx = this.sfx.ctx;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, time);
    osc.frequency.exponentialRampToValueAtTime(freq * 0.82, time + 0.22);
    gain.gain.setValueAtTime(0.0001, time);
    gain.gain.linearRampToValueAtTime(volume, time + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.001, time + 0.26);
    osc.connect(gain);
    gain.connect(this.bus);
    osc.start(time);
    osc.stop(time + 0.3);
  }

  airTick(time, volume) {
    this.scheduledNotes++;
    const ctx = this.sfx.ctx;
    const src = ctx.createBufferSource();
    const filter = ctx.createBiquadFilter();
    const gain = ctx.createGain();
    src.buffer = this.sfx.noiseBuf;
    filter.type = 'highpass';
    filter.frequency.value = 4800;
    gain.gain.setValueAtTime(0.0001, time);
    gain.gain.linearRampToValueAtTime(volume, time + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.001, time + 0.045);
    src.connect(filter);
    filter.connect(gain);
    gain.connect(this.bus);
    src.start(time);
    src.stop(time + 0.06);
  }
}

export const music = new Music();
