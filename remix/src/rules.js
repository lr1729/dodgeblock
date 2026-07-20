export const DEFAULT_RUN_RULES = Object.freeze({
  checkpoints: true,
  autoGuard: true,
});

export function normalizeRunRules(rules = {}) {
  rules = rules ?? {};
  return Object.freeze({
    checkpoints: rules.checkpoints !== false,
    autoGuard: rules.autoGuard !== false,
  });
}

export function runModeKey(rules) {
  return normalizeRunRules(rules).autoGuard ? 'guard' : 'hardcore';
}
