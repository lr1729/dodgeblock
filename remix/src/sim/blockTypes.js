import {
  BLOCK_FALL_CAP,
  BLOCK_GRAVITY,
} from '../constants.js';

const DEFAULTS = {
  variants: 1,
  gravity: BLOCK_GRAVITY,
  maxFallSpeed: BLOCK_FALL_CAP,
  canSupport: true,
  lethal: true,
  frameFor: null,
};

export const BLOCK_TYPES = new Map();

function define(id, spec = {}) {
  const full = { ...DEFAULTS, ...spec, id };
  BLOCK_TYPES.set(id, full);
  return full;
}

// Baseline permanent terrain. A falling piece can be vetoed with Focus, but
// once it lands its placement remains consequential for the rest of the run.
define('wood', { variants: 3 });

// Nonlethal incoming terrain. It can still push, trap, support, and be ridden.
define('gravel', {
  variants: 2,
  lethal: false,
});

// A 90px bridge. Its width naturally creates roofs and alternate supports;
// support otherwise follows the same visible-overlap rule as every material.
define('beam', {
  gravity: BLOCK_GRAVITY,
  maxFallSpeed: BLOCK_FALL_CAP,
});
