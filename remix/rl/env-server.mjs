import { Sim } from '../src/sim/sim.js';
import {
  ACTION_COUNT,
  OBS_SIZE,
  STATE_SIZE,
  actionInput,
  encodeObservation,
} from './env.mjs';

function arg(name, fallback) {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : Number(process.argv[index + 1]);
}

const count = arg('--envs', 128);
const baseSeed = arg('--seed', 1) >>> 0;
const hardcore = process.argv.includes('--hardcore');
const gridsBytes = count * OBS_SIZE;
const statesBytes = count * STATE_SIZE * 4;
const packetBytes = gridsBytes + statesBytes + count * (4 + 1 + 4 + 4 + 4);
const envs = Array.from({ length: count }, (_, index) => ({
  sim: new Sim((baseSeed + index * 0x9e3779b9) >>> 0, { rules: { autoGuard: !hardcore } }),
  focusHeld: false,
  jumpHeld: false,
  episodeReturn: 0,
  episodeLength: 0,
  episode: 0,
}));

function reset(entry, index) {
  entry.episode++;
  entry.sim = new Sim((baseSeed + index * 0x9e3779b9 + entry.episode * 0x85ebca6b) >>> 0, {
    rules: { autoGuard: !hardcore },
  });
  entry.focusHeld = false;
  entry.jumpHeld = false;
  entry.episodeReturn = 0;
  entry.episodeLength = 0;
}

function writePacket(rewards = null, dones = null, returns = null, lengths = null, heights = null) {
  const packet = Buffer.allocUnsafe(packetBytes);
  let stateOffset = gridsBytes;
  for (let index = 0; index < count; index++) {
    const { grid, state } = encodeObservation(envs[index].sim);
    Buffer.from(grid.buffer, grid.byteOffset, grid.byteLength).copy(packet, index * OBS_SIZE);
    for (let j = 0; j < STATE_SIZE; j++) packet.writeFloatLE(state[j], stateOffset + (index * STATE_SIZE + j) * 4);
  }
  let offset = gridsBytes + statesBytes;
  for (let i = 0; i < count; i++) packet.writeFloatLE(rewards?.[i] ?? 0, offset + i * 4);
  offset += count * 4;
  for (let i = 0; i < count; i++) packet[offset + i] = dones?.[i] ?? 0;
  offset += count;
  for (let i = 0; i < count; i++) packet.writeFloatLE(returns?.[i] ?? 0, offset + i * 4);
  offset += count * 4;
  for (let i = 0; i < count; i++) packet.writeUInt32LE(lengths?.[i] ?? 0, offset + i * 4);
  offset += count * 4;
  for (let i = 0; i < count; i++) packet.writeFloatLE(heights?.[i] ?? 0, offset + i * 4);
  process.stdout.write(packet);
}

function step(actions) {
  const rewards = new Float32Array(count);
  const dones = new Uint8Array(count);
  const returns = new Float32Array(count);
  const lengths = new Uint32Array(count);
  const heights = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const entry = envs[i];
    const beforeHeight = entry.sim.height;
    const selected = Math.min(ACTION_COUNT - 1, actions[i]);
    const mapped = actionInput(selected, entry.focusHeld, entry.jumpHeld);
    entry.focusHeld = mapped.focusHeld;
    entry.jumpHeld = mapped.jumpHeld;
    entry.sim.step(mapped.input);
    const reward = 0.01 + Math.max(0, entry.sim.height - beforeHeight) / 100 - (entry.sim.dead ? 1 : 0);
    entry.episodeReturn += reward;
    entry.episodeLength++;
    rewards[i] = reward;
    if (!entry.sim.dead) continue;
    dones[i] = 1;
    returns[i] = entry.episodeReturn;
    lengths[i] = entry.episodeLength;
    heights[i] = entry.sim.height;
    reset(entry, i);
  }
  writePacket(rewards, dones, returns, lengths, heights);
}

writePacket();
let pending = Buffer.alloc(0);
process.stdin.on('data', (chunk) => {
  pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
  while (pending.length >= count) {
    const actions = pending.subarray(0, count);
    pending = pending.subarray(count);
    step(actions);
  }
});
