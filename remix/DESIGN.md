# DodgeBlock Design Contract

## Thesis

The original is distinctive because the hazard constructs the level. Each
block must be read twice: first as an incoming collision and then as a permanent
terrain decision. Survival comes from managing several uncertain piles,
preserving routes, and leaving a locally safe position before it becomes a
trap.

The redesign adds agency inside that loop instead of placing unrelated rewards
on top of it. Height is the only score. Randomness creates terrain problems;
movement, timing, and deliberate structural cuts determine how well the player
answers them.

Randomness does not need to guarantee a win. It must avoid immediate situations
where normal perception and movement cannot matter. As in a strong action
roguelike, a skilled player should usually attribute a loss to a chain of
decisions even though exceptionally hostile runs remain possible.

## Simulation Invariants

The block simulation has four gameplay states:

1. A committed forecast with immutable position, width, and material.
2. A falling block with continuous vertical motion.
3. A fixed block occupying discrete grid-aligned terrain.
4. A faulted fixed block that remains solid until an atomic branch shatter.

The following are correctness requirements, not tuning preferences:

- fixed block rectangles never overlap;
- a falling block uses swept vertical contact and cannot tunnel through a
  surface crossed during one step;
- player movement resolves one axis at a time against the earliest crossed face
  and projects exactly to that surface;
- any positive horizontal overlap is physical contact;
- stable block identity is an immutable ID, never array order;
- in-flight blocks resolve lower-to-upper with stable ID tie-breaking;
- all materials use the same gravity and terminal fall speed;
- ordinary storm spawns reserve vertical separation from overlapping in-flight
  blocks;
- structural debris never re-enters gameplay physics;
- seeded input produces a deterministic history.

A general rigid-body solver is deliberately not used. Rotation, impulses,
solver jitter, and approximate stacks would add ambiguity to a game whose
terrain must be readable and replayable.

## Terrain

Blocks are 60 by 40 px and use 15 px horizontal anchors. This creates partial
shoulders and irregular routes without free-form subpixel stacks. A 15 px
contact is narrow but physically real. Hidden minimum-support thresholds were
removed because allowing a visible block to pass through such a contact looked
like broken collision.

All materials fall identically. Their meaning begins at contact:

| Material | Gameplay role |
| --- | --- |
| Wood | Lethal incoming baseline and permanent terrain |
| Gravel | Nonlethal incoming hazard and ordinary settled terrain |
| Beam | Wider terrain that creates broad ceilings and extra support contacts |

The bag contains 12 wood, 3 gravel, and 1 beam after the first four wood drops.
This bounds material droughts without scheduling a rescue at a useful location.

## Player Movement

Horizontal starts, reversals, and stops use separate curves: four frames to top
ground speed, seven for a full reversal, and three to stop. Held jump repeats on
valid footing. A wall jump accepts a fresh buffered jump press, held jump with
movement away from the wall, or a fresh press toward it; passive contact alone
never triggers one. Jump buffering and coyote time make late input reliable. A
wall jump can be used once from a given wall until the player lands or reaches
the opposite wall.
Solid contacts resolve to exact faces, so alignment never depends on the
fractional position of the previous frame. A descending player who clips a ledge
by at most six pixels is corrected into an open gap only when horizontal movement
already points there.

A lethal block kills only when its descending lower face reaches the player's
head with at least six pixels of horizontal overlap. Side contact and shallower
shoulder grazes push. Incidental top contact with a moving block is resolved as
ordinary support and grants no special boost. This makes contact direction,
rather than material intersection alone, determine the outcome.

Valid fixed footing awards new-height recharge immediately, so held auto-hop
does not lose progress. Jumping remains forgiving through six-frame ground
coyote time, seven-frame input buffering, four-frame wall grace, and bounded
upward corner correction.

## Focus

The player starts with three visible Focus charges. Activating one spends it immediately
and enters Aim:

- player, storm, blocks, gravity, and camera continue at 10% speed;
- steering, jumping, and collision remain active;
- an eight-way arrow can be adjusted throughout Aim;
- releasing a steering key preserves the last nonzero Aim direction;
- Aim lasts at most 90 real simulation ticks, or 1.5 seconds;
- release or timeout commits the dash;
- recharge and recharge milestones cannot advance during Aim.

The local cyan countdown ring makes the remaining decision time explicit.
There is no indefinitely optimal slow-motion mode.

The dash is directional and collision-limited. Material changes the danger or
shape of a block, never whether an otherwise identical cut input works:

- hitting one falling block shatters that block and ends the dash;
- hitting fixed wood, gravel, or beam from any direction marks its dependent
  branch for structural removal;
- ground and arena rails stop the dash.

Focus targeting uses swept contact time rather than block storage order. When
contacts are simultaneous, greater contact coverage wins before stable ID is
used as a final deterministic tie-breaker. The preview uses the same query as
the committed dash.

Focus is therefore one scarce answer with several uses: remove an immediate
hazard, escape a pocket, cross a route, or prune a tower.

Auto Guard is a configurable form of the same resource, not a separate life
system. A lethal overhead impact consumes one available charge automatically,
shatters every block in that same-frame crush incident, and kicks the player
upward. If Aim is active, its already-spent charge becomes the guard;
the player is never charged twice. Camera falls remain lethal. Disabling Auto
Guard produces the Hardcore ruleset, with a separate best height.

## Structural Cuts

There is no spontaneous load simulation. Hidden strength propagation produced
unattributable collapses and turned connected grid terrain into unstable
independent projectiles.

Instead, structural change begins only with an explicit player cut. Fixed
blocks share this rule in every direction so the player can reason from shape
and support rather than memorize material exceptions:

1. Find the target and every fixed block that would lack any alternate support
   if that target disappeared.
2. Preview the dependent branch during Aim, then mark it amber for 12 world
   frames after the committed player hit.
3. Keep every marked block fully solid during the warning.
4. Reconcile blocks that gain alternate support or join the dependency while
   the warning is active.
5. Remove the live dependent branch in one topology transaction.
6. Emit one `branchShatter` event containing snapshots for cosmetic debris.

No branch member becomes a falling simulation body. This gives tower pruning a
predictable macro effect while eliminating phasing, reversed landings, repeated
reseating, and accidental structural deaths.

## Progression

Score records the highest stable footing reached, not the peak of a jump. This
keeps score and Focus recharge aligned around the same objective: establish a
higher route.

One charge is earned for each new three-layer stable threshold. Progress made
while charges are full is discarded, so keeping protection ready sacrifices
future recharge progress. The HUD shows three charge diamonds and the remaining
new layers required for the next charge; there is no reserve.

The rising camera makes waiting on solved low terrain a losing strategy. Its
pressure follows the storm rate, keeping climbing and survival coupled.

When enabled, an exact in-memory checkpoint is armed every 100 stable height.
It can only be captured from safe footing outside Aim or a dash. Continuing is
explicitly assisted and cannot update either one-life best.

## Storm Fairness

The director intentionally does not solve the game. A full movement search was
expensive, could hitch the fixed update, and only proved the existence of a
quantized frame-perfect route that a person might never perceive.

The replacement is a conservative local veto. It estimates arrival time,
allows a human reaction margin, computes ordinary lateral reach, combines
near-synchronous lethal intervals, and considers only immediate walls or
obvious tall shaft closures. A candidate is rejected only when those intervals
cover the entire locally reachable range.

The heuristic does not prove long-term safety, optimize lanes, inject recovery
terrain, model Focus, or repair earlier strategic mistakes. Random terrain and
unfavorable future choices remain the source of tension.

Warning strips stack when their horizontal spans overlap. Brightness and arrow
depth represent estimated time to the readable field, so a block reserved far
above another drop remains visible without looking immediately imminent.

## Difficulty

Gravity stays bounded and readable. Difficulty rises through:

- the natural deposition rate;
- Build, Surge, Release, and Calm pacing;
- increased local placement correlation during Surge;
- camera pressure;
- the player's accumulated terrain decisions.

The system should be tuned only after physics invariants hold. Faster incorrect
physics is not higher difficulty.

## Verification

`npm test` covers movement, directional crush behavior, moving contact, Focus
timing and recharge, deterministic storm histories, material bags, committed
forecasts, local crush rejection, partial support, atomic branch shattering,
zero fixed overlap in long seeded runs, and director performance budgets.

Browser validation must cover desktop and landscape mobile, start/middle/end
Focus frames, a highlighted complete branch, cosmetic debris after collision is
gone, forecast readability, nonblank canvas output, and console errors.

Automated checks cannot establish fun. Playtests should record whether players
can explain deaths, distinguish material danger and geometry at a glance,
intentionally choose between Focus uses, cross between piles, and improve on
repeated seeds.
