function overlapsX(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x;
}

export function estimateFallFrames(distance, velocity, gravity, maxSpeed) {
  let remaining = Math.max(0, distance);
  let speed = Math.max(0, velocity);
  const acceleration = Math.max(0.0001, gravity);
  const cap = Math.max(speed, maxSpeed);
  let frames = 0;

  while (remaining > 0 && speed < cap) {
    speed = Math.min(cap, speed + acceleration);
    remaining -= speed;
    frames++;
  }
  if (remaining > 0) frames += Math.ceil(remaining / Math.max(0.0001, cap));
  return frames;
}

export function warningUrgency(source, screenY, clearY, pulse = 0) {
  const eta = estimateFallFrames(
    clearY - screenY,
    source.yVel ?? 0,
    source.spec?.gravity ?? 0.3,
    source.spec?.maxFallSpeed ?? 13,
  );
  const proximity = Math.max(0, Math.min(1, 1 - eta / 90));
  return {
    eta,
    progress: Math.min(1, 0.12 + proximity * 0.78 + pulse * 0.1),
  };
}

export function stackWarnings(items, { top = 3, gap = 11, maxDepth = 48 } = {}) {
  if (!items.length) return [];

  const ordered = items
    .map((item, inputIndex) => ({ ...item, inputIndex }))
    .sort((a, b) => a.eta - b.eta || a.x - b.x || a.inputIndex - b.inputIndex);
  const parent = ordered.map((_, index) => index);

  const find = (index) => {
    let root = index;
    while (parent[root] !== root) root = parent[root];
    while (parent[index] !== index) {
      const next = parent[index];
      parent[index] = root;
      index = next;
    }
    return root;
  };
  const union = (a, b) => {
    const rootA = find(a);
    const rootB = find(b);
    if (rootA !== rootB) parent[rootB] = rootA;
  };

  for (let i = 0; i < ordered.length; i++) {
    for (let j = i + 1; j < ordered.length; j++) {
      if (overlapsX(ordered[i], ordered[j])) union(i, j);
    }
  }

  const groups = new Map();
  for (let i = 0; i < ordered.length; i++) {
    const root = find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(ordered[i]);
  }

  const result = [];
  for (const group of groups.values()) {
    const rowGap = group.length > 1
      ? Math.min(gap, maxDepth / (group.length - 1))
      : 0;
    for (let i = 0; i < group.length; i++) {
      result.push({
        ...group[i],
        stackIndex: i,
        stackSize: group.length,
        y: top + (group.length - 1 - i) * rowGap,
      });
    }
  }

  return result.sort((a, b) => a.y - b.y || a.x - b.x || a.eta - b.eta);
}
