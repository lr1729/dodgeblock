// Persist rule preferences and one-life best heights for each protection mode.

import { DEFAULT_RUN_RULES, normalizeRunRules, runModeKey } from './rules.js';

const KEY = 'dodgeblock-remix-v3';
const PREVIOUS_KEY = 'dodgeblock-remix-v2';
const LEGACY_KEY = 'dodgeblock-remix-v1';
const LEGACY_RUN_RULES = Object.freeze({ checkpoints: true, autoGuard: false });

const DEFAULTS = Object.freeze({
  version: 3,
  bestHeight: 0,
  bestHeights: { guard: 0, hardcore: 0 },
  rules: DEFAULT_RUN_RULES,
});

function normalizedHeight(value) {
  const height = Number(value);
  return Number.isFinite(height) ? Math.max(0, Math.round(height)) : 0;
}

function readData(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const storedHeight = Number(parsed?.bestHeight);
    if (!Number.isFinite(storedHeight)) return null;
    return {
      bestHeight: normalizedHeight(storedHeight),
      bestHeights: {
        guard: normalizedHeight(parsed?.bestHeights?.guard),
        hardcore: normalizedHeight(parsed?.bestHeights?.hardcore ?? storedHeight),
      },
      rules: parsed?.rules === undefined
        ? LEGACY_RUN_RULES
        : normalizeRunRules(parsed.rules),
    };
  } catch {
    return null;
  }
}

function load() {
  const current = readData(KEY);
  if (current !== null) {
    return { ...DEFAULTS, ...current };
  }

  const previous = readData(PREVIOUS_KEY);
  const legacy = readData(LEGACY_KEY);
  const data = {
    ...DEFAULTS,
    bestHeight: previous?.bestHeight ?? legacy?.bestHeight ?? DEFAULTS.bestHeight,
    bestHeights: previous?.bestHeights ?? legacy?.bestHeights ?? {
      guard: 0,
      hardcore: previous?.bestHeight ?? legacy?.bestHeight ?? 0,
    },
    rules: previous?.rules ?? legacy?.rules ?? DEFAULT_RUN_RULES,
  };

  // Preserve previous results immediately; old blobs remain untouched.
  if (previous !== null || legacy !== null) {
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
  bestForRules(rules) {
    return normalizedHeight(this.data.bestHeights?.[runModeKey(rules)]);
  }

  recordRun({ height, seed: _seed, rules }) {
    const runHeight = normalizedHeight(height);
    const mode = runModeKey(rules);
    const newBest = runHeight > this.bestForRules(rules);
    if (newBest) {
      this.data.bestHeights = { ...this.data.bestHeights, [mode]: runHeight };
      this.data.bestHeight = Math.max(this.data.bestHeight, runHeight);
      this.save();
    }
    return newBest;
  }

  setRules(rules) {
    this.data.rules = normalizeRunRules(rules);
    this.save();
    return this.data.rules;
  }
}

export const storage = new Storage();
