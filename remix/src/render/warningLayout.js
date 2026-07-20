function overlapsX(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x;
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
