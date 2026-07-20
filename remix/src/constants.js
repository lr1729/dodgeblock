// The simulation runs at a fixed 60 Hz. Physics values are pixels per step.
export const STEP_MS = 1000 / 60;
export const MAX_STEPS_PER_FRAME = 5;

export const GAME_W = 800;
export const GAME_H = 500;
export const RES =
  typeof window !== 'undefined'
    ? Math.min(
        4,
        Math.max(
          1,
          Math.ceil((window.screen.width * (window.devicePixelRatio || 1)) / GAME_W),
        ),
      )
    : 1;

// Player movement. Fast reversal and input forgiveness make a late reaction a
// movement problem, not a fight against momentum.
export const GRAVITY = 0.34;
export const JUMP_VEL = 8.5;
export const PLAYER_FALL_CAP = 10;
export const MOVE_SPEED = 6;
export const MOVE_ACCEL_GROUND = 1.5;
export const MOVE_ACCEL_AIR = 1.2;
export const MOVE_TURN_GROUND = 2.4;
export const MOVE_TURN_AIR = 1.7;
export const MOVE_DECEL = 2;
export const JUMP_BUFFER_FRAMES = 7;
export const COYOTE_FRAMES = 6;
export const WALL_JUMP_X = 5.4;
export const WALL_JUMP_Y = 8.1;
export const WALL_COYOTE_FRAMES = 4;
export const CORNER_CORRECTION_PX = 4;

export const PLAYER_SIZE = 30;
export const PLAYER_START_X = 385;
export const PLAYER_START_Y = 270;

// A fully visible field with quarter-block placement. Partial horizontal
// support is the source of ledges, overhangs, ridges, and distinct pile shapes.
export const ARENA_X = 10;
export const BLOCK_W = 60;
export const BLOCK_H = 40;
export const SPAWN_GRID = BLOCK_W / 4;
export const ARENA_W = 780;
export const ARENA_COLS = ARENA_W / SPAWN_GRID;
export const PLAYER_MIN_X = ARENA_X;
export const PLAYER_MAX_X = ARENA_X + ARENA_W;
export const SPAWN_MIN_X = ARENA_X;
export const SPAWN_MAX_X = ARENA_X + ARENA_W - BLOCK_W;
export const GROUND = { x: ARENA_X, y: 300, w: ARENA_W, h: 900 };

// Ordinary wood enters the screen with close to the original game's urgency.
// The cap removes the original late-run singularity without turning the storm
// into slow lane markers.
export const BLOCK_GRAVITY = 0.3;
export const BLOCK_FALL_CAP = 13;
export const BLOCK_SPAWN_ABOVE = 270;
export const SQUISH_VEL = 3;
export const COLLAPSE_WARNING_FRAMES = 60;
export const CARVE_WARNING_FRAMES = 12;

// Focus: a short whole-world slow-motion aim followed by one committed burst.
// Charges return only through stable upward progress.
export const FOCUS_AIM_WORLD_SCALE = 0.1;
export const FOCUS_AIM_MAX_FRAMES = 90;
export const FOCUS_FRAMES = 8;
export const FOCUS_SPEED = 10;
export const FOCUS_DASH_WORLD_SCALE = 0.55;
export const FOCUS_EXIT_SPEED = 3.8;
export const FOCUS_CAP = 2;
export const FOCUS_RECHARGE_LAYERS = 3;

export const CHECKPOINT_HEIGHT = 1200;

// Camera pressure makes height and survival the same objective.
export const CAMERA_ANCHOR_Y = 160;
export const CAMERA_RISE_BASE = 0.022;
export const CAMERA_RATE_DIVISOR = 25;

// The storm breathes, but its underlying rate rises throughout a human-length
// run. The square-root tail prevents a solved hard plateau without recreating
// the original's runaway linear spawn singularity.
export const PHASES = [
  { id: 'calm', frames: 480, from: 0.82, to: 0.82 },
  { id: 'build', frames: 480, from: 0.82, to: 1.16 },
  { id: 'surge', frames: 360, from: 1.32, to: 1.32 },
  { id: 'release', frames: 240, from: 0.98, to: 0.72 },
];
export const OPENING_FRAMES = 360;
export const SPAWN_RATE_START = 1;
export const SPAWN_RATE_BODY = 4.2;
export const SPAWN_RATE_TAU_SECONDS = 75;
export const SPAWN_RATE_TAIL = 0.12;
export const SPAWN_ATTEMPTS = 16;
export const ARRIVAL_CLUSTER_FRAMES = 9;
export const TELEGRAPH_FRAMES = 18;
// Material changes what landed terrain means, not the ordering of blocks in
// flight. A shared fall profile prevents later drops from overtaking earlier
// ones in the same lane.

// Sky and terrain palette. Materials are visually distinct because each has a
// different tactical meaning.
export const COLOR_BG_GAME = 0x9fd6ee;
export const COLOR_SKY_TOP = 0x75bde0;
export const COLOR_SKY_BOTTOM = 0xe4f3f4;
export const COLOR_DEAD_SKY_TOP = 0x56616c;
export const COLOR_DEAD_SKY_BOTTOM = 0xa8afb4;
export const COLOR_BLOCK_FILLS = [0xd8a060, 0xcc9255, 0xe1aa69];
export const COLOR_BLOCK_BORDER = 0x67452b;
export const COLOR_BLOCK_TOP = 0xf3ce91;
export const COLOR_BLOCK_SHADE = 0x865a35;
export const COLOR_WARNING = 0xe8483f;
export const COLOR_GRASS = 0x4da866;
export const COLOR_GRASS_DARK = 0x32774a;
export const COLOR_SOIL_TOP = 0x795238;
export const COLOR_SOIL_BOTTOM = 0x402c25;
export const COLOR_PLAYER = 0xe8433f;
export const COLOR_PLAYER_BORDER = 0x972d2a;
export const COLOR_PLAYER_MOUTH = 0x581f20;
export const COLOR_FOCUS = 0x37d6d0;
export const COLOR_GRAVEL = 0x8d918b;
export const COLOR_BEAM = 0x425466;
export const FONT = "'Trebuchet MS', Verdana, Arial, sans-serif";
