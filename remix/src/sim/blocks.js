import {
  ARENA_COLS,
  ARENA_W,
  ARENA_X,
  BLOCK_H,
  BLOCK_SPAWN_ABOVE,
  BLOCK_W,
  CARVE_WARNING_FRAMES,
  COLLAPSE_WARNING_FRAMES,
  GROUND,
  SPAWN_GRID,
} from '../constants.js';
import { BLOCK_TYPES } from './blockTypes.js';
import { stackContact } from './util.js';

const CONTACT_EPSILON = 0.001;
const SPAWN_GAP = 2;

function overlapsX(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x;
}

function overlapsStrict(a, b) {
  return (
    overlapsX(a, b) &&
    a.y < b.y + b.h &&
    a.y + a.h > b.y
  );
}

function uniqueBlocks(blocks) {
  return [...new Set(blocks)].sort((a, b) => a.id - b.id);
}

export class Block {
  constructor(x, y, w = BLOCK_W, h = BLOCK_H, shade = 0, type = 'wood') {
    this.id = -1;
    this.idx = -1;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.yVel = 0;
    this.xVel = 0;
    this.fixed = false;
    this.shade = shade;
    this.type = type;
    this.spec = BLOCK_TYPES.get(type);
    this.fixedAtFrame = -1;
    this.impactVel = 0;
    this.spawnFrame = -1;
    this.zone = 0;
    this.faultTimer = 0;
    this.faultReason = null;
    this.faultRoot = null;
    this.faultDuration = 0;
  }
}

export class BlockManager {
  constructor(sim) {
    this.sim = sim;
    this.blocks = [];
    this.falling = [];
    this.layers = [[]];
    this.faults = [];
    this.nextBlockId = 1;
    this.topologyVersion = 0;
  }

  get length() {
    return this.blocks.length;
  }

  snapshot() {
    return {
      blocks: this.blocks.map((block) => ({
        id: block.id,
        idx: block.idx,
        x: block.x,
        y: block.y,
        w: block.w,
        h: block.h,
        yVel: block.yVel,
        xVel: block.xVel,
        fixed: block.fixed,
        shade: block.shade,
        type: block.type,
        fixedAtFrame: block.fixedAtFrame,
        impactVel: block.impactVel,
        spawnFrame: block.spawnFrame,
        zone: block.zone,
        faultTimer: block.faultTimer,
        faultReason: block.faultReason,
        faultRootId: block.faultRoot?.id ?? null,
        faultDuration: block.faultDuration,
        faultBranchIds: (block.faultBranch ?? []).map((member) => member.id),
      })),
      fallingIds: this.falling.map((block) => block.id),
      faultIds: this.faults.map((block) => block.id),
      nextBlockId: this.nextBlockId,
      topologyVersion: this.topologyVersion,
    };
  }

  restore(state) {
    const byId = new Map();
    this.blocks = state.blocks.map((data) => {
      const block = new Block(data.x, data.y, data.w, data.h, data.shade, data.type);
      const { faultRootId: _root, faultBranchIds: _branch, ...values } = data;
      Object.assign(block, values);
      byId.set(block.id, block);
      return block;
    });
    for (const data of state.blocks) {
      const block = byId.get(data.id);
      block.faultRoot = byId.get(data.faultRootId) ?? null;
      if (data.faultBranchIds.length) {
        block.faultBranch = data.faultBranchIds.map((id) => byId.get(id)).filter(Boolean);
      }
    }
    this.falling = state.fallingIds.map((id) => byId.get(id)).filter(Boolean);
    this.faults = state.faultIds.map((id) => byId.get(id)).filter(Boolean);
    this.nextBlockId = state.nextBlockId;
    this.topologyVersion = state.topologyVersion;
    this.rebuildLayers();
    return byId;
  }

  xForColumn(col, width = BLOCK_W) {
    const maxCol = Math.floor((GROUND.w - width) / SPAWN_GRID);
    return ARENA_X + Math.max(0, Math.min(maxCol, col)) * SPAWN_GRID;
  }

  columnForX(x) {
    return Math.max(
      0,
      Math.min(ARENA_COLS - 1, Math.floor((x - ARENA_X) / SPAWN_GRID)),
    );
  }

  spawnColumn(col, camY, type = 'wood', opts = {}) {
    const width = opts.w ?? BLOCK_W;
    const x = this.xForColumn(col, width);
    let y = opts.y ?? -camY - BLOCK_SPAWN_ABOVE;
    if (opts.reserve) y = this.reserveSpawnY(x, y, width, opts.h ?? BLOCK_H);
    return this.spawnAt(x, y, type, { ...opts, y, w: width });
  }

  reserveSpawnY(x, y, w, h) {
    let reserved = y;
    for (const other of this.falling) {
      if (!overlapsX({ x, w }, other)) continue;
      reserved = Math.min(reserved, other.y - h - SPAWN_GAP);
    }
    return reserved;
  }

  spawnAt(x, y, type = 'wood', opts = {}) {
    const spec = BLOCK_TYPES.get(type);
    if (!spec) throw new Error(`unknown block type: ${type}`);
    const b = new Block(
      x,
      y,
      opts.w ?? BLOCK_W,
      opts.h ?? BLOCK_H,
      opts.shade ?? this.sim.rng.int(0, spec.variants),
      type,
    );
    b.yVel = opts.yVel ?? 0;
    b.zone = this.sim.director?.zoneIndex ?? 0;
    b.spawnFrame = this.sim.frame;
    this.add(b);
    return b;
  }

  add(b) {
    b.id = this.nextBlockId++;
    b.idx = b.id;
    this.blocks.push(b);
    this.falling.push(b);
    this.sim.events.emit('blockSpawn', b);
  }

  layerFor(layer) {
    while (this.layers.length <= layer) this.layers.push([]);
    return this.layers[layer];
  }

  fixAt(b, topLayer) {
    const x = ARENA_X + SPAWN_GRID * Math.round((b.x - ARENA_X) / SPAWN_GRID);
    const y = GROUND.y - BLOCK_H * topLayer;
    const placed = { x, y, w: b.w, h: b.h };
    const collision = this.blocks.find(
      (other) => other !== b && other.fixed && overlapsStrict(placed, other),
    );
    if (collision) {
      throw new Error(`illegal fixed overlap: block ${b.id} with ${collision.id}`);
    }

    b.x = x;
    b.y = y;
    b.impactVel = b.yVel;
    b.fixed = true;
    b.yVel = 0;
    b.xVel = 0;
    b.fixedAtFrame = this.sim.frame;
    const rows = Math.max(1, Math.round(b.h / BLOCK_H));
    for (let layer = topLayer - rows + 1; layer <= topLayer; layer++) {
      if (layer < 1) continue;
      const list = this.layerFor(layer);
      if (!list.includes(b)) list.push(b);
    }
    this.topologyVersion++;
    this.sim.events.emit('blockFix', b);
  }

  update(timeScale = 1) {
    this.updateFaults(timeScale);

    const oldY = new Map(this.falling.map((b) => [b, b.y]));
    this.falling.sort((a, b) => {
      const bottomA = oldY.get(a) + a.h;
      const bottomB = oldY.get(b) + b.h;
      return bottomB - bottomA || a.id - b.id;
    });

    const resolved = [];
    for (let i = 0; i < this.falling.length; i++) {
      const b = this.falling[i];
      const previousY = oldY.get(b);
      const previousBottom = previousY + b.h;
      b.xVel = 0;
      b.yVel = Math.min(b.spec.maxFallSpeed, b.yVel + b.spec.gravity * timeScale);
      const proposedY = b.y + b.yVel * timeScale;
      const proposedBottom = proposedY + b.h;

      const fixedTop = this.firstFixedSurface(b, previousBottom, proposedBottom);
      if (fixedTop !== null) {
        const rows = Math.max(1, Math.round(b.h / BLOCK_H));
        const supportLayer = Math.round((GROUND.y - fixedTop) / BLOCK_H);
        this.fixAt(b, supportLayer + rows);
        this.falling.splice(i--, 1);
        resolved.push(b);
        continue;
      }

      let movingSupport = null;
      for (const lower of resolved) {
        if (lower.fixed || !overlapsX(b, lower)) continue;
        const oldTop = oldY.get(lower);
        const newTop = lower.y;
        const alreadyOverlapping =
          previousY < oldTop + lower.h && previousBottom > oldTop;
        if (
          alreadyOverlapping ||
          (
            previousBottom <= oldTop + CONTACT_EPSILON &&
            proposedBottom >= newTop - CONTACT_EPSILON
          )
        ) {
          movingSupport = lower;
          break;
        }
      }

      if (movingSupport) {
        b.y = movingSupport.y - b.h;
        b.yVel = movingSupport.yVel;
      } else {
        b.y = proposedY;
      }
      resolved.push(b);
    }
  }

  firstFixedSurface(b, previousBottom, proposedBottom) {
    let top = null;
    if (b.y < GROUND.y && proposedBottom >= GROUND.y) {
      top = GROUND.y;
    }
    for (const other of this.blocks) {
      if (other === b || !other.fixed || !overlapsX(b, other)) continue;
      if (
        b.y < other.y &&
        proposedBottom >= other.y - CONTACT_EPSILON &&
        (top === null || other.y < top)
      ) {
        top = other.y;
      }
    }
    return top;
  }

  updateFaults(timeScale) {
    for (let i = 0; i < this.faults.length; i++) {
      const root = this.faults[i];
      root.faultTimer = Math.max(0, root.faultTimer - timeScale);
      const previousBranch = root.faultBranch ?? [];
      const branch = this.dependentBranch(root);
      root.faultBranch = uniqueBlocks(branch);
      const nextBranch = new Set(root.faultBranch);
      for (const block of previousBranch) {
        if (nextBranch.has(block) || block === root) continue;
        block.faultTimer = 0;
        block.faultReason = null;
        block.faultRoot = null;
        block.faultDuration = 0;
      }
      for (const block of root.faultBranch) {
        block.faultTimer = root.faultTimer;
        block.faultDuration = root.faultDuration;
        block.faultReason = root.faultReason;
        block.faultRoot = root;
      }
      if (root.faultTimer > 0) continue;
      this.faults.splice(i--, 1);
      this.shatterBranch(root);
    }
  }

  remove(b, { deferLayerCleanup = false } = {}) {
    const i = this.blocks.indexOf(b);
    if (i === -1) return false;
    this.blocks.splice(i, 1);
    const fallingIndex = this.falling.indexOf(b);
    if (fallingIndex !== -1) this.falling.splice(fallingIndex, 1);
    const faultIndex = this.faults.indexOf(b);
    if (faultIndex !== -1) this.faults.splice(faultIndex, 1);
    if (b.fixed && !deferLayerCleanup) this.detachFromLayers(b);
    b.fixed = false;
    b.faultTimer = 0;
    b.faultReason = null;
    b.faultRoot = null;
    b.faultDuration = 0;
    this.topologyVersion++;
    return true;
  }

  detachFromLayers(b) {
    for (const list of this.layers) {
      const i = list.indexOf(b);
      if (i !== -1) list.splice(i, 1);
    }
  }

  rebuildLayers() {
    this.layers = [[]];
    for (const b of this.blocks) {
      if (!b.fixed) continue;
      const rows = Math.max(1, Math.round(b.h / BLOCK_H));
      const topLayer = Math.round((GROUND.y - b.y) / BLOCK_H);
      for (let layer = topLayer - rows + 1; layer <= topLayer; layer++) {
        if (layer >= 1) this.layerFor(layer).push(b);
      }
    }
  }

  removeFixedAndCollapse(target) {
    if (!target.fixed) {
      const snapshot = { ...target };
      this.remove(target);
      this.sim.events.emit('blockShatter', { block: snapshot });
      return [target];
    }
    if (!this.markFault(target, 'carve')) return [];
    return target.faultRoot?.faultBranch ?? target.faultBranch ?? [];
  }

  markFault(root, reason = 'carve') {
    if (!root.fixed) return false;
    const branch = uniqueBlocks(this.dependentBranch(root));
    if (!branch.length) return false;
    const branchSet = new Set(branch);
    const overlapping = this.faults.filter((fault) =>
      (fault.faultBranch ?? []).some((block) => branchSet.has(block)),
    );
    if (overlapping.length) {
      const covered = new Set(overlapping.flatMap((fault) => fault.faultBranch ?? []));
      if (branch.every((block) => covered.has(block))) return true;
      if (![...covered].every((block) => branchSet.has(block))) return false;

      const requestedTimer = reason === 'carve'
        ? CARVE_WARNING_FRAMES
        : COLLAPSE_WARNING_FRAMES;
      const timer = Math.min(
        requestedTimer,
        ...overlapping.map((fault) => fault.faultTimer),
      );
      this.faults = this.faults.filter((fault) => !overlapping.includes(fault));
      for (const block of covered) {
        block.faultTimer = 0;
        block.faultReason = null;
        block.faultRoot = null;
        block.faultDuration = 0;
      }
      root.faultTimer = timer;
      root.faultDuration = Math.max(timer, CARVE_WARNING_FRAMES);
      root.faultReason = reason;
      root.faultRoot = root;
      root.faultBranch = branch;
      for (const block of branch) {
        block.faultTimer = timer;
        block.faultDuration = root.faultDuration;
        block.faultReason = reason;
        block.faultRoot = root;
      }
      this.faults.push(root);
      this.sim.events.emit('blockFaultMerge', { block: root, root, blocks: branch, reason });
      return true;
    }

    root.faultDuration = reason === 'carve'
      ? CARVE_WARNING_FRAMES
      : COLLAPSE_WARNING_FRAMES;
    root.faultTimer = root.faultDuration;
    root.faultReason = reason;
    root.faultRoot = root;
    root.faultBranch = branch;
    for (const block of branch) {
      block.faultTimer = root.faultTimer;
      block.faultDuration = root.faultDuration;
      block.faultReason = reason;
      block.faultRoot = root;
    }
    this.faults.push(root);
    this.sim.events.emit('blockFault', { block: root, root, blocks: branch, reason });
    return true;
  }

  shatterBranch(root) {
    const liveBranch = uniqueBlocks(this.dependentBranch(root));
    const snapshots = liveBranch.map((b) => ({
      id: b.id,
      x: b.x,
      y: b.y,
      w: b.w,
      h: b.h,
      type: b.type,
      shade: b.shade,
      zone: b.zone,
    }));
    for (const block of liveBranch) this.remove(block, { deferLayerCleanup: true });
    for (const block of root.faultBranch ?? []) {
      if (liveBranch.includes(block)) continue;
      block.faultTimer = 0;
      block.faultReason = null;
      block.faultRoot = null;
      block.faultDuration = 0;
    }
    this.rebuildLayers();
    this.sim.events.emit('branchShatter', { root, blocks: snapshots });
    return liveBranch;
  }

  unsupportedBlocks(excluded = new Set()) {
    const unsupported = new Set(excluded);
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of this.blocks) {
        if (!b.fixed || unsupported.has(b)) continue;
        if (this.isSupported(b, unsupported)) continue;
        unsupported.add(b);
        changed = true;
      }
    }
    return [...unsupported].filter((b) => b.fixed);
  }

  dependentBranch(root) {
    if (!root || !root.fixed || !this.blocks.includes(root)) return [];
    return this.unsupportedBlocks(new Set([root]));
  }

  isSupported(b, excluded = new Set()) {
    const rows = Math.max(1, Math.round(b.h / BLOCK_H));
    const topLayer = Math.round((GROUND.y - b.y) / BLOCK_H);
    const supportLayer = topLayer - rows;
    if (supportLayer <= 0) return true;
    return (this.layers[supportLayer] ?? []).some(
      (support) =>
        support !== b &&
        !excluded.has(support) &&
        support.fixed &&
        support.spec.canSupport &&
        stackContact(b, support),
    );
  }

  surfaceLayers() {
    const surface = new Array(ARENA_COLS).fill(0);
    for (const b of this.blocks) {
      if (!b.fixed) continue;
      const topLayer = Math.round((GROUND.y - b.y) / BLOCK_H);
      for (let col = 0; col < ARENA_COLS; col++) {
        const center = ARENA_X + col * SPAWN_GRID + SPAWN_GRID / 2;
        if (center >= b.x && center < b.x + b.w) {
          surface[col] = Math.max(surface[col], topLayer);
        }
      }
    }
    return surface;
  }

  isExposedTop(b) {
    if (!b.fixed) return false;
    const topLayer = Math.round((GROUND.y - b.y) / BLOCK_H);
    return !(this.layers[topLayer + 1] ?? []).some(
      (other) => other !== b && other.fixed && overlapsX(b, other),
    );
  }
}
