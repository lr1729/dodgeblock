export function heightReward({
  beforeHeight,
  afterHeight,
  blockHeight,
  dead,
  deathPenalty,
  aliveReward,
  worldScale,
}) {
  return Math.max(0, afterHeight - beforeHeight) / blockHeight +
    (dead ? 0 : aliveReward * worldScale) -
    (dead ? deathPenalty : 0);
}

export function targetReward({
  beforeHeight,
  afterHeight,
  targetHeight,
  discount,
  worldScale,
  dead,
  success,
}) {
  const beforePotential = Math.min(1, beforeHeight / targetHeight);
  const afterPotential = dead || success ? 0 : Math.min(1, afterHeight / targetHeight);
  return (success ? 1 : 0) +
    Math.pow(discount, worldScale) * afterPotential -
    beforePotential;
}
