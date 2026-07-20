// The remix only persists the one result that matters: how high you climbed.

const KEY = 'dodgeblock-remix-v2';
const LEGACY_KEY = 'dodgeblock-remix-v1';

const DEFAULTS = Object.freeze({
  version: 2,
  bestHeight: 0,
});

function normalizedHeight(value) {
  const height = Number(value);
  return Number.isFinite(height) ? Math.max(0, Math.round(height)) : 0;
}

function readBestHeight(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const storedHeight = Number(JSON.parse(raw)?.bestHeight);
    return Number.isFinite(storedHeight) ? normalizedHeight(storedHeight) : null;
  } catch {
    return null;
  }
}

function load() {
  const currentBest = readBestHeight(KEY);
  if (currentBest !== null) {
    return { ...DEFAULTS, bestHeight: currentBest };
  }

  const legacyBest = readBestHeight(LEGACY_KEY);
  const data = {
    ...DEFAULTS,
    bestHeight: legacyBest ?? DEFAULTS.bestHeight,
  };

  // Preserve a v1 best immediately; the old blob remains untouched.
  if (legacyBest !== null) {
    try {
      localStorage.setItem(KEY, JSON.stringify(data));
    } catch {
      /* private browsing */
    }
  }

  return data;
}

class Storage {
  constructor() {
    this.data = load();
  }

  save() {
    try {
      localStorage.setItem(KEY, JSON.stringify(this.data));
    } catch {
      /* private browsing */
    }
  }

  // The seed stays in the run contract for replay/debugging, but is not meta.
  recordRun({ height, seed: _seed }) {
    const runHeight = normalizedHeight(height);
    const newBest = runHeight > this.data.bestHeight;
    if (newBest) {
      this.data.bestHeight = runHeight;
      this.save();
    }
    return newBest;
  }
}

export const storage = new Storage();
