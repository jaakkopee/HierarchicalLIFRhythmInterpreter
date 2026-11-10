# LIFCellPolyRhythmWrapper `tune()` Design Plan (Brainstorm)

## Goal

Make each `LIFCell` fire at a **tempo-relative rate** defined by the current `POLYRHYTHM_MATRIX` schema:

- one `"meter"` label per hypothesis (for example `"7/8"`)
- beat-indexed entries (`1..N`) each with:
  - `"length"` (timing multiplier)
  - `"weight"` (accent/probability weight)

while still allowing expressive variation and cross-rhythm behavior.

---

## 1) Define the timing model first

Treat one global clock as authoritative:

- `tempo_bpm` (global)
- `base_subdivision` (for example eighth-note)
- `dt` (simulation step size in seconds)

Derived timing:

- `seconds_per_beat = 60 / tempo_bpm`
- `base_unit_seconds = seconds_per_beat * base_subdivision_factor`
- `target_period_seconds = base_unit_seconds * length`

Where `length` comes from each beat entry (for example `16/7`, `16/4`, etc.).

---

## Matrix-specific parsing rules (updated)

For each hypothesis dictionary in `POLYRHYTHM_MATRIX`:

1. Read and keep `"meter"` as metadata.
2. Collect only integer keys (`1..N`) as beat slots.
3. For each beat slot, read:
   - `length`: tempo-relative duration multiplier
   - `weight`: beat accent strength (currently values like `1`, `1/8`, `1/16`)
4. Normalize weights per hypothesis so they are directly usable as spike-bias multipliers.

Important current property of your matrix:

- within one meter row, all beat slots share the same `length`
- rhythmic character is mainly shaped by the per-slot `weight` accents

---

## 2) Separate responsibilities in the wrapper

`LIFCellPolyRhythmWrapper` should own:

1. **Target timing** (period/frequency) from rhythm metadata  
2. **Cell parameter adaptation** (how to move LIF dynamics toward target)  
3. **Phase state** (where this cell is inside its rhythmic cycle)  
4. **Spike quality metrics** (did it fire at desired times?)

This avoids mixing tuning logic into high-level rhythm detection.

---

## 3) Map musical timing -> LIF behavior

Use a two-layer control strategy:

### A. Slow calibration (per pattern update)

When `tune(params)` is called:

- Compute each beat slot’s `target_period_seconds`
- Convert to `target_steps = target_period_seconds / dt`
- Compute desired tonic drive / threshold / leak regime for that period
- Store those values as this cell’s current targets

### B. Fast phase-locked modulation (per simulation step)

At each `lif_step`:

- Advance wrapper phase with global clock
- Compute phase error vs expected beat phase
- Modulate input current (or threshold bias) so firing probability peaks near target phase
- Reduce excitation outside target phase window

This makes firing periodic and tempo-locked instead of purely random.

---

## 4) Practical control knobs per cell

For each wrapped cell, keep these conceptual controls:

- `target_rate_hz` (from period)
- `excitability_bias` (base current offset)
- `phase_window_width` (how tight the beat lock is)
- `refractory_guard` (minimum gap between spikes)
- `accent_gain` (strong vs weak beat emphasis)

Use hypothesis `weight` values as weights:

- higher `weight` => higher `accent_gain`, narrower phase window
- lower `weight` => lower gain, wider window, lower trigger probability

---

## 5) Tempo-relative adaptation loop

Track observed inter-spike intervals (ISI) per cell:

- `observed_period = moving_average(ISI)`
- `period_error = target_period - observed_period`

Adjust slowly:

- if firing too fast: lower excitability or raise threshold
- if firing too slow: raise excitability or lower threshold

Use low adaptation gain to avoid oscillation (musically: “settle, don’t jitter”).

---

## 6) Phase-based gating for polyrhythm clarity

For polyrhythms, each cell gets its own phase accumulator:

- `phase += dt / target_period`
- wrap phase to `[0, 1)`

Gate shape idea:

- strong gate near phase `0` (beat onset)
- weak gate elsewhere
- optional secondary lobe for syncopation

This yields stable 3:2 / 4:3 style interference patterns when multiple cells run at different periods.

---

## 7) Hypothesis-driven weighting

Since each beat now has explicit `weight`:

- use normalized `weight` to scale gate amplitude
- use `weight` to bias spike probability around each beat phase
- optionally use low `weight` beats as syncopation candidates (wider, softer gates)
- optionally blend multiple hypotheses by confidence

So matrix `weight` values directly influence **how strongly** each beat is expressed, while `length` controls **when** beats recur.

---

## 8) Multi-timescale architecture alignment

Match your three layers:

- **MusicalForm**: updates coarse section context (slowly changes priors/weights)
- **RhythmicStructure**: owns tempo-locked wrapped LIF cells and phase logic
- **LowLevelEventGenerator**: converts spikes to events with optional density/chaos post-filter

Keep tuning in RhythmicStructure; keep event ornamentation in LowLevelEventGenerator.

---

## 9) Validation criteria (no code, measurable behavior)

Check these metrics:

1. **Period accuracy**: mean ISI close to target period for each tuned cell  
2. **Phase locking**: spike-time concentration around target phase  
3. **Polyrhythm stability**: expected composite cycle repeats over long runs  
4. **Tempo tracking**: when BPM changes, cells retune smoothly within N beats  
5. **Musical robustness**: no permanent silence, no uncontrolled burst lock

---

## 10) Rollout plan

1. Implement wrapper state model (targets, phase, ISI stats)  
2. Add tune-time conversion from `length` to target period and `weight` to accent gain  
3. Add per-step phase gate modulation around LIF input/threshold  
4. Add slow adaptive correction from period error  
5. Add per-beat `weight` handling from hypothesis dictionaries  
6. Evaluate with fixed tempos and tempo ramps  
7. Calibrate adaptation/gating constants by listening + metrics

---

## Key design principle

Treat LIF cells as **stochastic oscillators under phase-constrained drive**:  
the rhythm definition sets target periods/phases, and the wrapper nudges each cell so spike probability peaks at musically correct times.
