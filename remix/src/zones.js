// Altitude bands are presentation only. They may change sky, ambience, block
// tint, and music, but never spawn rules, scoring, controls, or physics.
export const ZONES = [
  {
    id: 'meadow',
    name: 'MEADOW',
    threshold: 0,
    skyTop: 0x75bde0,
    skyBottom: 0xe4f3f4,
    blockTint: 0xffffff,
    musicIntensity: 0,
    cloudAlpha: 0.72,
    rain: false,
    stars: false,
    aurora: false,
  },
  {
    id: 'stormfront',
    name: 'STORM LINE',
    threshold: 900,
    skyTop: 0x536878,
    skyBottom: 0xb8c9cf,
    blockTint: 0xe4e8e8,
    musicIntensity: 1,
    cloudAlpha: 0.42,
    rain: true,
    stars: false,
    aurora: false,
  },
  {
    id: 'cloudtop',
    name: 'CLOUDTOP',
    threshold: 2400,
    skyTop: 0x9fcfe2,
    skyBottom: 0xf4d9b4,
    blockTint: 0xfff3dc,
    musicIntensity: 2,
    cloudAlpha: 0.9,
    rain: false,
    stars: false,
    aurora: false,
  },
  {
    id: 'aurora',
    name: 'AURORA',
    threshold: 4800,
    skyTop: 0x2d3142,
    skyBottom: 0x746d83,
    blockTint: 0xdce8df,
    musicIntensity: 3,
    cloudAlpha: 0.12,
    rain: false,
    stars: true,
    aurora: true,
  },
];

export function zoneIndexFor(camY) {
  const altitude = Math.max(0, Number.isFinite(camY) ? camY : 0);
  for (let i = ZONES.length - 1; i >= 0; i--) {
    if (altitude >= ZONES[i].threshold) return i;
  }
  return 0;
}
