# DodgeBlock Remix

DodgeBlock is Tetris from inside the pile. Every falling block is first a
hazard and then persistent terrain. The only objective is stable height:
survive, shape useful routes, and keep climbing before the rising camera
removes the bottom of the level.

The redesign preserves uncertain pile management while giving the player
reliable answers to bad terrain. It has no near-miss score, random powerups,
periodic shield, or secondary reward economy.

## Controls

| Action | Keyboard | Touch |
| --- | --- | --- |
| Move | `A` / `D` or Left / Right | Hold the lower-left or lower-right side |
| Jump / auto-hop | `W`, Up, or `Space` | Tap or hold the upper area |
| Fast fall | `S` or Down | - |
| Focus Aim | Hold `Shift`, `X`, or `K` | Swipe and keep holding |
| Commit Focus | Release the Focus key | Release the swipe |
| Mute | `M` | - |

Focus slows the whole simulation to 10% speed for at most 1.5 real seconds.
The player continues moving and can adjust the eight-way aim direction. Release
early, or let the timer expire, to commit the dash.

## Core Mechanics

- The bounded 780 px arena uses 15 px horizontal block anchors. Partial
  overlaps create ledges, overhangs, shoulders, and competing piles.
- All materials share one fall profile, so later blocks cannot overtake slower
  ones. Any positive horizontal contact is physical support; blocks never
  phase through ledges or settle into occupied space.
- **Wood** is the lethal baseline. Only a descending overhead impact kills.
  Side contact pushes; incidental top contact with a moving block uses the same
  collision rules but grants no special jump or reward.
- **Gravel** is the nonlethal incoming material. It still pushes, traps,
  and supports terrain, but a direct overhead impact does not kill. Its danger
  difference is visible before contact.
- **Beams** are 90 px wide blocks. Their broader roofs and support contacts
  change terrain without adding special hidden rules.
- **Focus** starts with two charges. With Auto Guard enabled, a charge
  automatically absorbs a lethal overhead crush and removes every block in that
  same impact incident. Hitting a falling block shatters that one
  hazard. Hitting settled terrain from any of the eight directions marks its
  dependent branch for removal.
- Cutting a fixed block previews its dependent branch during Aim, then keeps it
  amber and solid for 12 world frames before it disappears
  atomically. Debris is cosmetic and cannot collide, kill, or reseat.
- Wall jumps recover from pockets and pile edges without requiring a random
  double-jump pickup. Wall contact includes four-frame jump grace, and small
  upward head-corner catches are corrected without cancelling the jump.
- The HUD states how many new stable layers remain before the next dash. Focus
  recharges on the landing frame after progress through each three-layer
  threshold. Progress made while charges are full is discarded, and progress
  cannot advance during Aim.
- The storm commits position, width, and material 18 frames before the drop.
  A cheap local safety heuristic rejects only obvious synchronized crushes;
  most terrain remains random and strategically imperfect.
- Difficulty comes from rising density, camera pressure, and correlated storm
  phases rather than unbounded gravity or unreadable physics.
- Every 1,200 height arms an in-memory checkpoint when the run rule is enabled.
  Continuing from the death
  screen restores the exact simulation state and marks the run as assisted, so
  it cannot replace the one-life best. Auto Guard and Hardcore keep separate
  best heights.

The home screen exposes Checkpoints and Auto Guard as persistent run rules. Auto
Guard makes Focus charges double as health; Hardcore preserves one-hit crushes.

See [`DESIGN.md`](./DESIGN.md) for the simulation contract and design rationale.

## Development

```sh
npm install
npm run dev
npm test
npm run build
```

Use `?seed=N` for a repeatable run and `?test` to expose the browser test bridge.
The authoritative simulation runs at fixed 60 Hz in `src/sim/`.
