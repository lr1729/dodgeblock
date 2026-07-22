#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import numpy as np


ACTION_COUNT = 18
TERRAIN_ROWS = 32
TERRAIN_COLS = 52
TERRAIN_SIZE = TERRAIN_ROWS * TERRAIN_COLS
SKYLINE_SIZE = 52
FALLING_COUNT = 32
FALLING_FEATURES = 11
FORECAST_COUNT = 16
FORECAST_FEATURES = 10
STATE_SIZE = 32


class EnvWorker:
    def __init__(self, count, seed, archive_probability=0.0, archive_capacity=2048):
        server = Path(__file__).with_name('env-server-v2.mjs')
        command = [
            'node', str(server), '--envs', str(count), '--seed', str(seed),
            '--archive-probability', str(archive_probability),
            '--archive-capacity', str(archive_capacity),
        ]
        self.count = count
        self.observation_bytes = (
            TERRAIN_SIZE * 2 + SKYLINE_SIZE +
            (FALLING_COUNT * FALLING_FEATURES +
             FORECAST_COUNT * FORECAST_FEATURES + STATE_SIZE) * 4
        )
        self.packet_size = count * self.observation_bytes + count * (4 + 1 + 4 + 4 + 4 + 4 + 4 + 4 + 4)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _read_exact(self):
        data = bytearray(self.packet_size)
        view = memoryview(data)
        offset = 0
        while offset < len(data):
            size = self.process.stdout.readinto(view[offset:])
            if not size:
                raise RuntimeError(f'environment server exited with {self.process.poll()}')
            offset += size
        return data

    def read(self):
        data = self._read_exact()
        count = self.count
        offset = 0

        terrain = np.frombuffer(data, '<u2', count * TERRAIN_SIZE, offset)
        terrain = terrain.reshape(count, TERRAIN_ROWS, TERRAIN_COLS).copy()
        offset += count * TERRAIN_SIZE * 2

        skyline = np.frombuffer(data, np.uint8, count * SKYLINE_SIZE, offset)
        skyline = skyline.reshape(count, SKYLINE_SIZE).copy()
        offset += count * SKYLINE_SIZE

        falling_size = count * FALLING_COUNT * FALLING_FEATURES
        falling = np.frombuffer(data, '<f4', falling_size, offset)
        falling = falling.reshape(count, FALLING_COUNT, FALLING_FEATURES).copy()
        offset += falling_size * 4

        forecast_size = count * FORECAST_COUNT * FORECAST_FEATURES
        forecasts = np.frombuffer(data, '<f4', forecast_size, offset)
        forecasts = forecasts.reshape(count, FORECAST_COUNT, FORECAST_FEATURES).copy()
        offset += forecast_size * 4

        state = np.frombuffer(data, '<f4', count * STATE_SIZE, offset)
        state = state.reshape(count, STATE_SIZE).copy()
        offset += count * STATE_SIZE * 4

        rewards = np.frombuffer(data, '<f4', count, offset).copy()
        offset += count * 4
        dones = np.frombuffer(data, np.uint8, count, offset).copy()
        offset += count
        returns = np.frombuffer(data, '<f4', count, offset).copy()
        offset += count * 4
        lengths = np.frombuffer(data, '<u4', count, offset).copy()
        offset += count * 4
        heights = np.frombuffer(data, '<f4', count, offset).copy()
        offset += count * 4
        episode_starts = np.frombuffer(data, '<f4', count, offset).copy()
        offset += count * 4
        start_heights = np.frombuffer(data, '<f4', count, offset).copy()
        offset += count * 4
        current_heights = np.frombuffer(data, '<f4', count, offset).copy()
        offset += count * 4
        world_scales = np.frombuffer(data, '<f4', count, offset).copy()

        return {
            'terrain': terrain,
            'skyline': skyline,
            'falling': falling,
            'forecasts': forecasts,
            'state': state,
            'rewards': rewards,
            'dones': dones,
            'returns': returns,
            'lengths': lengths,
            'heights': heights,
            'episode_starts': episode_starts,
            'start_heights': start_heights,
            'current_heights': current_heights,
            'world_scales': world_scales,
        }

    def send(self, actions):
        self.process.stdin.write(np.asarray(actions, np.uint8).tobytes())
        self.process.stdin.flush()

    def close(self):
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class ParallelEnvBridge:
    def __init__(self, workers, envs_per_worker, seed, archive_probability=0.0, archive_capacity=2048):
        self.workers = [
            EnvWorker(
                envs_per_worker,
                seed + index * 0x1F123BB5,
                archive_probability,
                archive_capacity,
            )
            for index in range(workers)
        ]
        self.count = workers * envs_per_worker
        self.executor = ThreadPoolExecutor(max_workers=workers)

    @staticmethod
    def _merge(packets):
        return {key: np.concatenate([packet[key] for packet in packets], axis=0) for key in packets[0]}

    def read(self):
        futures = [self.executor.submit(worker.read) for worker in self.workers]
        return self._merge([future.result() for future in futures])

    def step(self, actions):
        chunks = np.split(np.asarray(actions, np.uint8), len(self.workers))
        for worker, chunk in zip(self.workers, chunks):
            worker.send(chunk)
        return self.read()

    def close(self):
        for worker in self.workers:
            worker.close()
        self.executor.shutdown(wait=True)
