
## v16 — the ladder promotes past rung 600 for the first time (2026-07-26)

Seeded from repeat-4-s8, training at action repeat 4 and gating at 60 Hz.

| | det | per-layer | climb | median |
|---|---|---|---|---|
| seed checkpoint, on the gate | 0.317 | 0.9262 | 11.2 h/s | 440 |
| **after 8M more frames (rung-600-x0)** | **0.471** | **0.9510** | **13.8 h/s** | **560** |
| control, same gate protocol | 0.344 | 0.9313 | ~10.5 h/s | 440 |

**PROMOTE -> rung 850.** The v6 ladder self-terminated at rung 600 with
NEEDS-ATTENTION; its refine midpoint of 550 fell inside 1.15x of the last pass, so
it stopped. This is the first time any configuration has cleared that gate.

Hazard at rung 600 falls 0.0687 -> 0.0490, a **1.40x reduction = 0.34 nats** --
roughly double what the 8M A/B measured on its own (0.16), so extending training on
top of action repeat keeps paying rather than saturating at 3M frames the way every
previous rung did.

Climb rate 10.5 -> 13.8 h/s, **+31%**. Three discount strengths spanning a 3.5x
range of time preference were run specifically to buy speed and moved it by
nothing. A change to *when the policy acts* bought it, with the objective never
mentioning time.

Caveats, since this is one run on a 512-episode gate: 0.471 against a control of
0.344 is far outside the eval noise on that protocol, but the gate is a quarter the
size of the 2048-episode standard adopted earlier today, and the seeded rung's own
0.317 versus the matrix's 0.4097 for the same checkpoint shows how much that gate
moves. The promotion decision is safe -- 0.471 clears the 0.50 promote band on
per-layer terms and is nowhere near the boundary -- but the *magnitude* should be
read from the 2048-episode numbers, not from these.

What this does not yet show: rung 850 is still the easy regime. Saturated per-layer
survival for this policy class was 0.7393 against the 0.9567 needed. Whether the
ladder keeps climbing or stalls at 850-1000 like every previous attempt is the
thing to watch, and it is now running unattended.

### The wall moved but did not break (2026-07-26)

| rung | frames | det | per-layer | outcome |
|---|---|---|---|---|
| 600 | 8M | 0.471 | 0.9510 | PROMOTE -> 850 |
| 850 | 8M | 0.281 | 0.9420 | EXTEND |
| 850 | +8M | 0.289 | 0.9433 | **REFINE -> 700** |

The ladder stalled at 850 and refined back down. Same qualitative wall as v6, moved
up one rung: v6 stopped at 600, this stops at 850, **+42% in target height** for a
1.21x hazard reduction at the stall point (0.9313 -> 0.9433, 0.19 nats).

**And it corrects last hour's entry.** I wrote that training on top of action repeat
"keeps paying rather than saturating at 3M frames the way every previous rung did",
on the strength of rung 600 going 0.317 -> 0.471 over an extension. Rung 850's
extension bought +0.008 det and +0.0013 per-layer -- saturated, exactly like every
previous rung. So the claim held at 600 and fails at 850, which means it was a
property of that rung being easy for this policy, not a property of action repeat.
Registering it because it was stated one entry earlier as a general finding.

The pattern across the whole project is now consistent and worth naming plainly:
**every intervention moves the wall a little and none removes it.** Action repeat is
the largest mover so far -- 0.19 nats at the stall point, against 1.80 still needed
in the saturated regime -- and it still leaves a wall of the same shape, at a rung
where extension stops paying and the driver refines downward.
