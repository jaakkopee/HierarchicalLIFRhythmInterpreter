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

- `auto` or blank line: run automated input sequence
- `auto N`: run auto mode for `N` steps
- `density X`: set density knob (`0.0..1.0`)
- `chaos X`: set chaos knob (`0.0..1.0`)
- `knobs D C`: set both knobs
- `beatstrength X`: set beat accent strength (`0.0..2.0`)
- `audiofollow on|off [mix]`: enable/disable audio-driven chaos+density (`mix` in `0..1`)
- `scale NAME`: set scale
- `scale NAME ROOT`: set scale and root MIDI note (`0..127`)
- `scales`: list available scales
- `show`: print current knob and scale settings
- `q` / `quit` / `exit`: stop

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
- `density` mainly affects activity level and gate duty.
- `chaos` mainly affects randomness and pattern variability.
- `beatstrength` controls how strongly detected beats accent network drive.
- `audiofollow` maps audio energy to density and spectral flux to chaos.

## Quick starting presets

- Stable groove: `knobs 0.45 0.20`
- Evolving rhythm: `knobs 0.60 0.45`
- Glitchier texture: `knobs 0.75 0.80`
- Minor pentatonic bass area: `scale minor_pentatonic 36`
- Brighter modal feel: `scale lydian 48`
- GM percussion kit: `scale drums`

## Notes

- If you see no sound, verify your MIDI routing (DAW/synth input) and selected output port.
- The script emits MIDI events only; external synth/sampler/DAW must render audio.
