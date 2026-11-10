# Brainstorm: How many neurons are needed per rhythm in `POLYRHYTHM_MATRIX`?

## Scope

This is a design brainstorm for sizing the `RhythmicStructure` layer so each matrix rhythm can be represented reliably with `LIFCell` populations.

---

## Core idea

For each rhythm row, let:

- `S` = number of beat slots (integer keys `1..S` in that row)

Then choose one of these sizing models:

1. **Minimal (structural only):** `N = S`  
   - one neuron per slot
2. **Robust (recommended baseline):** `N = 2S`  
   - one primary neuron + one support neuron per slot (stability/recovery)
3. **Expressive (accent/detail):** `N = 3S`  
   - primary + support + ornament/syncopation neuron per slot

---

## Rhythm-by-rhythm neuron counts (from current matrix layout)

| Matrix row meter label | Slot count `S` | Minimal `N=S` | Robust `N=2S` | Expressive `N=3S` |
|---|---:|---:|---:|---:|
| 8/8 (16 slots) | 16 | 16 | 32 | 48 |
| 15/8 | 15 | 15 | 30 | 45 |
| 7/4 (14 slots) | 14 | 14 | 28 | 42 |
| 13/8 | 13 | 13 | 26 | 39 |
| 12/8 | 12 | 12 | 24 | 36 |
| 11/8 | 11 | 11 | 22 | 33 |
| 10/8 | 10 | 10 | 20 | 30 |
| 9/8 | 9 | 9 | 18 | 27 |
| 4/4 (8 slots) | 8 | 8 | 16 | 24 |
| 7/8 | 7 | 7 | 14 | 21 |
| 6/8 | 6 | 6 | 12 | 18 |
| 5/8 | 5 | 5 | 10 | 15 |
| 2/4 (4 slots row) | 4 | 4 | 8 | 12 |
| 3/8 | 3 | 3 | 6 | 9 |
| 2/4 (2 slots row) | 2 | 2 | 4 | 6 |
| 1/1 | 1 | 1 | 2 | 3 |

---

## What this means for global layer size

### If each cell can be retuned dynamically (shared pool)

- Required pool size is driven by the **largest row only**.
- Largest `S` is 16, so:
  - Minimal pool: 16
  - Robust pool: 32
  - Expressive pool: 48

### If all rhythms must coexist concurrently (parallel hypotheses)

- Sum of all slots in current matrix is `136`.
- So:
  - Minimal full-parallel: 136
  - Robust full-parallel: 272
  - Expressive full-parallel: 408

---

## Recommendation

For your current architecture, a practical sequence is:

1. Start with **shared-pool robust**: `32` neurons (for best rhythm at-a-time quality).
2. Move to `48` if accents/syncopations feel too weak.
3. Only use `136+` style allocations when you truly need simultaneous multi-hypothesis rendering instead of hypothesis switching/blending.

---

## Why these multipliers make sense

- `1x` captures the pattern shape but is fragile under noise.
- `2x` gives enough redundancy for phase locking and anti-silence behavior.
- `3x` enables stable core pulse plus independent accent/detail behavior per slot.

---

## Validation checklist (design-level)

For each rhythm row, verify:

1. Slot-level pulse appears at the intended tempo-relative period.
2. High-weight slots fire more reliably than low-weight slots.
3. Pattern survives tempo ramps (no total collapse or runaway bursting).
4. Transition between rows settles within a few beats.

