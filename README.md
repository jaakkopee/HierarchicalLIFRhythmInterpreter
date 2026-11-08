# HierarchicalLIFRhythmInterpreter

This project is a Norse-based rhythmic MIDI generator built around a small recurrent LIF network.

Current main script: `LIFMIDI.py`.

## What it does

- creates a 16-neuron LIF network (`norse.torch.LIFCell`)
- uses mixed excitatory/inhibitory recurrent connections
- drives the network with oscillatory + stochastic input
- emits MIDI note on/off events when neurons spike
- supports live control for rhythm density, chaos, and musical scale
- supports audio-reactive control where input audio drives density/stochasticity

## Requirements

- Python 3.10 or 3.11 recommended
- `torch`
- `norse`
- `numpy`
- `python-rtmidi` (or compatible `rtmidi` binding)

Install example:

```bash
pip install torch norse numpy python-rtmidi
```

## Running

```bash
python3 LIFMIDI.py
```

Optional: force a MIDI output by name match:

```bash
LIFNN_MIDI_PORT=IAC python3 LIFMIDI.py
```

On startup, the script prints available MIDI output ports and opens one.

## Interactive commands

- `auto` or blank line
- `auto N`
- `density X`
- `chaos X`
- `knobs D C`
- `beatstrength X`
- `audiofollow on|off [mix]`
- `scale NAME`
- `scale NAME ROOT`
- `scales`
- `show`
- `q` / `quit` / `exit`

You can also type a plain numeric value (for example `1.2`) to provide manual control input for one update step.

## Processing chain (Mermaid)

```mermaid
flowchart TD
    A[Mic input frame] --> B[Audio features<br/>RMS energy + spectral flux]
    M[Manual input value] --> C[Input blend & normalization]
    B --> C
    B --> D[audiofollow mapping]
    D --> E[density, chaos]
    B --> F[BeatTracker]
    F --> G[beat_now]
    H[beatstrength] --> I[beat_level]
    G --> I
    E --> J[Drive shaping<br/>phase speed, amp, noise, gate]
    C --> J
    E --> K[Recurrent gain]
    I --> L[Beat accent to tonic/gain]
    J --> N[Total neuron drive]
    K --> N
    L --> N
    N --> O[LIF neurons]
    O --> P[Spikes]
    P --> Q[Scale note mapping]
    Q --> R[MIDI note on/off]
```

## CLI parameter reference (usage, sensible values, chain effect)

### `density`

- Usage: `density X`
- Range: `0.0..1.0`
- Sensible values:
  - `0.35-0.50`: sparse, slower pulse openings
  - `0.50-0.70`: balanced groove
  - `0.70-0.85`: denser rhythms
- Processing-chain effect:
  - increases rhythmic gate duty cycle
  - increases tonic/base excitation
  - increases pulse probability in auto mode
  - slightly increases oscillation speed/amplitude coupling

### `chaos`

- Usage: `chaos X`
- Range: `0.0..1.0`
- Sensible values:
  - `0.15-0.30`: stable and repeatable
  - `0.30-0.55`: lively and evolving
  - `0.55-0.85`: noisy/glitchy
- Processing-chain effect:
  - increases stochastic noise injected into neuron drive
  - increases recurrent variability/intensity
  - increases auto-input drift and random pulse magnitude

### `knobs`

- Usage: `knobs D C`
- Example: `knobs 0.60 0.45`
- Sensible values:
  - start with `0.55 0.35`, then tweak one axis at a time
- Processing-chain effect:
  - shorthand that sets `density` and `chaos` together
  - immediately reshapes gate behavior, noise level, and recurrent feel

### `beatstrength`

- Usage: `beatstrength X`
- Range: `0.0..2.0`
- Sensible values:
  - `0.0`: disable beat accents
  - `0.6-1.1`: musical accents (recommended)
  - `1.2-1.8`: aggressive accents/pumping
- Processing-chain effect:
  - scales `beat_level` from beat detections
  - adds accent to tonic drive and global drive multiplier on beat hits
  - stronger values produce more spike bursts near detected transients

### `audiofollow`

- Usage: `audiofollow on|off [mix]`
- `mix` range: `0.0..1.0`
- Sensible values:
  - `audiofollow on 0.30`: subtle reaction
  - `audiofollow on 0.60`: balanced reactive control
  - `audiofollow on 0.80`: strong mic-driven modulation
- Processing-chain effect:
  - maps audio RMS energy -> density target
  - maps audio spectral flux -> chaos target
  - applies smoothing and blend by `mix`
  - with `off`, density/chaos stay manual

### `scale`

- Usage: `scale NAME` or `scale NAME ROOT`
- `ROOT` range: `0..127` (MIDI note number)
- Sensible values:
  - melodic: `scale minor_pentatonic 36`, `scale lydian 48`
  - drum mode: `scale drums` (GM kit on MIDI channel 10)
- Processing-chain effect:
  - remaps spike indices to MIDI notes
  - in `drums`, switches to GM percussion note layout and drum channel

### `auto`

- Usage: `auto` or `auto N`
- `N` sensible values:
  - `64-128`: quick test
  - `256-512`: phrase-length exploration
  - `1024+`: long generative run
- Processing-chain effect:
  - feeds the network with an internal stochastic control signal
  - control signal drift/pulses are shaped by `chaos` and `density`

## Scale support

Built-in scale names:

- `major`
- `minor`
- `harmonic_minor`
- `melodic_minor`
- `major_pentatonic`
- `minor_pentatonic`
- `blues`
- `dorian`
- `phrygian`
- `lydian`
- `mixolydian`
- `locrian`
- `whole_tone`
- `chromatic`
- `drums` (GM drum mapping on MIDI channel 10)

## Model behavior notes

- Very large manual inputs are compressed internally (`tanh`) to keep dynamics stable.
- The network has rhythmic gating, slow modulation, and anti-silence recovery to avoid dead states.
- Beat detection from audio transients can inject accents into the drive stage.

## Quick starting presets

- Stable groove: `knobs 0.45 0.20`
- Evolving rhythm: `knobs 0.60 0.45`
- Glitchier texture: `knobs 0.75 0.80`
- Minor pentatonic bass area: `scale minor_pentatonic 36`
- Brighter modal feel: `scale lydian 48`
- GM percussion kit: `scale drums`

## Live performance cheat sheet

Use these as quick command sequences during live tweaking.

### Scene 1: Tight groove (safe start)

```text
scale minor_pentatonic 36
knobs 0.48 0.22
beatstrength 0.75
audiofollow on 0.35
auto 256
```

### Scene 2: Musical evolution (balanced)

```text
scale dorian 38
knobs 0.60 0.45
beatstrength 1.00
audiofollow on 0.60
auto 384
```

### Scene 3: Drum-forward reactive jam

```text
scale drums
knobs 0.68 0.40
beatstrength 1.20
audiofollow on 0.70
auto 512
```

### Scene 4: Glitch burst

```text
scale whole_tone 48
knobs 0.78 0.82
beatstrength 1.55
audiofollow on 0.80
auto 192
```

### Scene 5: Ambient sparse texture

```text
scale lydian 50
knobs 0.35 0.18
beatstrength 0.45
audiofollow on 0.25
auto 512
```

### Fast rescue moves (if sound is too chaotic or too static)

- Too chaotic/noisy:
  - `chaos 0.25`
  - `density 0.50`
  - `beatstrength 0.80`
- Too static/boring:
  - `chaos 0.55`
  - `density 0.65`
  - `beatstrength 1.20`
- Mic influence too strong/weak:
  - strong: `audiofollow on 0.75`
  - subtle: `audiofollow on 0.30`
  - off: `audiofollow off`

## Notes

- If you see no sound, verify your MIDI routing (DAW/synth input) and selected output port.
- The script emits MIDI events only; external synth/sampler/DAW must render audio.
