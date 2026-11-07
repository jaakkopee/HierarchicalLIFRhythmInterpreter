import norse
import torch
from norse.torch import LIFCell
import os
import time

import numpy as np
import sounddevice as sd
import threading
import rtmidi

midi_out = rtmidi.RtMidiOut()

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "chromatic": list(range(12)),
    "drums": [],
}

GM_DRUM_NOTES = [
    36,  # Bass Drum 1
    38,  # Acoustic Snare
    42,  # Closed Hi-Hat
    46,  # Open Hi-Hat
    41,  # Low Floor Tom
    43,  # High Floor Tom
    45,  # Low Tom
    47,  # Low-Mid Tom
    48,  # Hi-Mid Tom
    50,  # High Tom
    49,  # Crash Cymbal 1
    51,  # Ride Cymbal 1
    39,  # Hand Clap
    37,  # Side Stick
    54,  # Tambourine
    56,  # Cowbell
]

# a LIF neural network with MIDI output.
class LIFNN:
    def __init__(self, num_cells=16):
        # Initialize the LIF neural network with the specified number of cells.
        self.lif_cells = [LIFCell() for _ in range(num_cells)]
        # Initialize the states and outputs for each LIF cell.
        self.states = [cell.initial_state(torch.zeros(1)) for cell in self.lif_cells]
        # output for each LIF cell. Initially set to zero.
        self.outputs = [torch.zeros(1) for _ in self.lif_cells]
        self.num_cells = len(self.lif_cells)
        # Initialize the input, lock, scale, root note, and MIDI output.
        self.input = torch.zeros(1)
        self.lock = threading.Lock() # Lock for thread-safe operations on the network.
        self.scale_name = "minor_pentatonic"
        self.root_note = 36
        self.notes = self._build_scale_notes(self.scale_name, self.root_note)
        self.melodic_channel = 0
        self.drum_channel = 9
        self.midi_channel = self.melodic_channel
        self.midi_out = midi_out
        self.active_notes = set()
        # Initialize the spike trace, recurrent weights, and other network parameters.
        self.spike_trace = torch.zeros(self.num_cells, dtype=torch.float32)
        self.trace_decay = 0.82
        self.recurrent_weights = self._build_alternating_weights()
        self.recurrent_gain = 0.42
        self.bias = torch.tensor(
            [0.08 if i % 2 == 0 else 0.03 for i in range(self.num_cells)],
            dtype=torch.float32,
        )
        self.phase = 0.0
        self.base_phase_step = 0.32
        self.mod_phase = 0.0
        self.mod_phase_step = 0.043
        self.step_index = 0
        self.silent_steps = 0
        self.rng = np.random.default_rng()
        self.auto_input = 1.0
        self.neuron_offsets = torch.linspace(-0.5, 0.5, self.num_cells, dtype=torch.float32)
        self.phase_offsets = torch.tensor(
            self.rng.uniform(0.0, 2.0 * np.pi, size=self.num_cells),
            dtype=torch.float32,
        )
        self.density = 0.58
        self.chaos = 0.36
        self._open_midi_output()
        self.audio_input = AudioInput()  # Initialize the audio input handler.
        self.latest_audio_input = np.zeros(1024)  # Initialize the latest audio input buffer.
        self.audio_input.start()  # Start capturing audio input.
    
    def _build_scale_notes(self, scale_name, root_note):
        if scale_name == "drums":
            notes = []
            idx = 0
            while len(notes) < self.num_cells:
                notes.append(GM_DRUM_NOTES[idx % len(GM_DRUM_NOTES)])
                idx += 1
            return notes

        intervals = SCALES[scale_name]
        notes = []
        octave = 0
        idx = 0
        # Build the scale notes based on the specified scale and root note.
        while len(notes) < self.num_cells:
            note = root_note + intervals[idx] + (12 * octave)
            notes.append(int(np.clip(note, 0, 127)))
            idx += 1
            # Move to the next interval in the scale.
            if idx >= len(intervals):
                idx = 0
                octave += 1 # Increment the octave after completing one cycle through the scale intervals.
        return notes

    def set_scale(self, scale_name=None, root_note=None):
        if scale_name is not None:
            normalized = scale_name.strip().lower().replace("-", "_")
            if normalized not in SCALES:
                available = ", ".join(sorted(SCALES.keys()))
                print(f"Unknown scale '{scale_name}'. Available scales: {available}")
                return
            self.scale_name = normalized
        if root_note is not None:
            self.root_note = int(np.clip(int(root_note), 0, 127))
        self.midi_channel = self.drum_channel if self.scale_name == "drums" else self.melodic_channel

        self.all_notes_off()
        self.notes = self._build_scale_notes(self.scale_name, self.root_note)
        print(
            f"Scale => {self.scale_name} root={self.root_note} "
            f"channel={self.midi_channel + 1} notes={self.notes}"
        )

    def set_knobs(self, density=None, chaos=None):
        if density is not None:
            self.density = float(np.clip(float(density), 0.0, 1.0))
        if chaos is not None:
            self.chaos = float(np.clip(float(chaos), 0.0, 1.0))
        print(f"Knobs => density={self.density:.2f} chaos={self.chaos:.2f}")
    # Build the alternating weights matrix for the network.
    def _build_alternating_weights(self):
        weights = torch.empty((self.num_cells, self.num_cells), dtype=torch.float32)
        for i in range(self.num_cells):
            for j in range(self.num_cells):
                if i == j:
                    weights[i, j] = -0.22 # Self-inhibition for each neuron.
                elif (i % 2) == (j % 2):
                    weights[i, j] = 0.28 # Excitatory connection for neurons with the same parity.
                else:
                    weights[i, j] = -0.38 # Inhibitory connection for neurons with different parity.
        return weights
    # Compute the oscillating drive for the network based on the input value.
    def _oscillating_drive(self, input_value):
        # Get the 16-band FFT representation of the input.
        input_fft = self.audio_input.get_input_fft_16bands()
        input_value = np.mean(input_fft)
        # Normalize the input value and compute the phase speed for oscillation.
        input_norm = float(torch.tanh(torch.tensor(float(input_value), dtype=torch.float32) / 6.0).item())
        phase_speed = self.base_phase_step * (1.0 + 0.8 * input_norm)
        phase_speed *= 0.75 + 0.55 * self.density
        phase_speed = float(np.clip(phase_speed, 0.08, 0.85))
        self.phase += phase_speed

        # Compute the oscillation amplitude and phase for each neuron.
        amp = 0.12 + (0.25 + 0.25 * self.density) * abs(input_norm)
        phase_vector = torch.full((self.num_cells,), self.phase, dtype=torch.float32)
        parity_phase = torch.tensor(
            [0.0 if i % 2 == 0 else np.pi for i in range(self.num_cells)],
            dtype=torch.float32,
        )
        oscillation = amp * torch.sin(phase_vector + parity_phase + self.phase_offsets)
        # Compute the base input, topology term, and noise for each neuron.
        base = torch.full((self.num_cells,), input_norm * (0.45 + 0.20 * self.density), dtype=torch.float32)
        topology_term = input_norm * 0.35 * self.neuron_offsets
        noise_std = 0.015 + 0.11 * self.chaos
        # Generate Gaussian noise for each neuron based on the chaos parameter.
        noise = torch.tensor(self.rng.normal(0.0, noise_std, size=self.num_cells), dtype=torch.float32)
        return base + topology_term + oscillation + noise

    def _next_auto_input(self):
        drift_std = 0.05 + 0.12 * self.chaos
        drift = float(self.rng.normal(0.0, drift_std))
        self.auto_input = float(np.clip(self.auto_input + drift, 0.4, 1.8))
        pulse = 0.0
        pulse_probability = 0.05 + 0.18 * self.density
        if self.rng.random() < pulse_probability:
            pulse = float(self.rng.uniform(0.15, 0.50 + 0.35 * self.chaos))
        return self.auto_input + pulse

    def _rhythmic_gate(self):
        gate = torch.zeros(self.num_cells, dtype=torch.float32)
        for i in range(self.num_cells):
            period = 5 + (i % 4)
            duty = 1 + int(round(self.density * 3.0)) + ((i // 3) % 2)
            duty = min(duty, period - 1)
            if (self.step_index + i) % period < duty:
                gate[i] = 1.0
        return gate
    
    def _open_midi_output(self):
        get_port_count = getattr(self.midi_out, "getPortCount", None)
        get_port_name = getattr(self.midi_out, "getPortName", None)
        open_port = getattr(self.midi_out, "openPort", None)

        if callable(get_port_count) and callable(open_port):
            port_count = get_port_count()
            if port_count > 0:
                port_names = []
                for idx in range(port_count):
                    name = f"port {idx}"
                    if callable(get_port_name):
                        try:
                            name = get_port_name(idx)
                        except rtmidi.Error:
                            pass
                    port_names.append(name)

                print("Available MIDI output ports:")
                for idx, name in enumerate(port_names):
                    print(f"  [{idx}] {name}")

                selected_port = None
                preferred_port_name = os.environ.get("LIFNN_MIDI_PORT")
                if preferred_port_name:
                    preferred_port_name_lower = preferred_port_name.lower()
                    for idx, name in enumerate(port_names):
                        if preferred_port_name_lower in name.lower():
                            selected_port = idx
                            break
                    if selected_port is None:
                        print(
                            f"Requested LIFNN_MIDI_PORT '{preferred_port_name}' not found; falling back to auto selection."
                        )

                if selected_port is None:
                    for idx, name in enumerate(port_names):
                        lowered = name.lower()
                        if "iac" in lowered or "bus" in lowered or "virtual" in lowered:
                            selected_port = idx
                            break

                if selected_port is None:
                    selected_port = 0

                open_port(selected_port)
                print(f"MIDI output opened: {port_names[selected_port]} (index {selected_port})")
                return

        open_virtual_port = getattr(self.midi_out, "openVirtualPort", None)
        if callable(open_virtual_port):
            open_virtual_port("LIFNN Output")
            print("MIDI virtual output opened: LIFNN Output")
            return

        print("Warning: no MIDI output port could be opened.")

    def _send_midi(self, status, note, velocity):
        message = [int(status), int(note), int(velocity)]
        last_error = None
        send_errors = (TypeError, rtmidi.Error)

        if hasattr(self.midi_out, "send_message"):
            try:
                self.midi_out.send_message(message)
                return
            except send_errors as exc:
                last_error = exc
            try:
                self.midi_out.send_message(bytes(message))
                return
            except send_errors as exc:
                last_error = exc

        if hasattr(self.midi_out, "sendMessage"):
            for candidate in (message, tuple(message), bytes(message), message[0] | (message[1] << 8) | (message[2] << 16)):
                try:
                    self.midi_out.sendMessage(candidate)
                    return
                except send_errors as exc:
                    last_error = exc

            midi_message_cls = getattr(rtmidi, "MidiMessage", None)
            if midi_message_cls is not None:
                midi_message = None
                if (message[0] & 0xF0) == 0x90 and message[2] > 0 and hasattr(midi_message_cls, "noteOn"):
                    midi_message = midi_message_cls.noteOn(0, message[1], message[2])
                elif ((message[0] & 0xF0) == 0x80 or ((message[0] & 0xF0) == 0x90 and message[2] == 0)) and hasattr(midi_message_cls, "noteOff"):
                    midi_message = midi_message_cls.noteOff(0, message[1])

                if midi_message is not None:
                    try:
                        self.midi_out.sendMessage(midi_message)
                        return
                    except send_errors as exc:
                        last_error = exc

        raise RuntimeError(
            f"Unsupported RtMidi output API for message {message}: {last_error}"
        )
    # Send a MIDI message with the specified status, note, and velocity.
    def note_on(self, note):
        # Trigger a note-on event for the specified note if it is not already active.
        with self.lock:
            if note not in self.active_notes:
                self._send_midi(0x90 | self.midi_channel, note, 127)
                self.active_notes.add(note)

    def note_off(self, note):
        # Trigger a note-off event for the specified note if it is currently active.
        with self.lock:
            if note in self.active_notes:
                self._send_midi(0x80 | self.midi_channel, note, 0)
                self.active_notes.remove(note)

    def all_notes_off(self):
        # Turn off all currently active notes.
        with self.lock:
            for note in list(self.active_notes):
                self._send_midi(0x80 | self.midi_channel, note, 0)
                self.active_notes.remove(note)

    def update(self, input_value):
        # Update the network state based on the input value and internal dynamics.
        
        # Normalize the input value and compute the external and recurrent drives.
        input_norm = float(torch.tanh(torch.tensor(float(input_value), dtype=torch.float32) / 6.0).item())
        # Compute the normalized input and the external oscillating drive.
        external_drive = self._oscillating_drive(input_value)
        # Compute the recurrent drive based on the recurrent weights and spike trace.
        recurrent_gain = self.recurrent_gain * (0.7 + 0.6 * self.chaos)
        recurrent_drive = recurrent_gain * torch.mv(self.recurrent_weights, self.spike_trace)
        # Update the modulation phase and compute the slow bias for each neuron.
        self.mod_phase += self.mod_phase_step * (1.0 + 0.35 * abs(input_norm))
        slow_bias = 0.16 * torch.sin(
            torch.full((self.num_cells,), self.mod_phase, dtype=torch.float32) + self.phase_offsets
        )
        # Compute the rhythmic gate and mix it with the drive.
        gate = self._rhythmic_gate()
        gate_mix = (0.24 + 0.45 * self.density) + (0.76 - 0.45 * self.density) * gate
        tonic = 0.05 + 0.20 * self.density + 0.12 * abs(input_norm)
        gated_drive = (external_drive + recurrent_drive + self.bias + slow_bias + tonic) * gate_mix
        # Apply the gated drive and implement the rescue mechanism for silent neurons.
        total_drive = gated_drive
        rescue_after = int(round(16 - 10 * self.density))
        if self.silent_steps >= rescue_after:
            rescue_idx = self.step_index % self.num_cells
            total_drive[rescue_idx] += 0.55
            total_drive[(rescue_idx + (self.num_cells // 2)) % self.num_cells] += 0.35

        current_spikes = torch.zeros(self.num_cells, dtype=torch.float32)
        for i, cell in enumerate(self.lif_cells):
            # Compute the input for the neuron and update its state.
            neuron_input = total_drive[i].unsqueeze(0)
            self.outputs[i], self.states[i] = cell(neuron_input, self.states[i])
            if self.outputs[i] > 0:
                # Trigger note-on if the neuron spikes.
                self.note_on(self.notes[i])
                current_spikes[i] = 1.0
            else:
                # Trigger note-off if the neuron is silent.
                self.note_off(self.notes[i])
        self.spike_trace = self.trace_decay * self.spike_trace + current_spikes
        if current_spikes.sum().item() > 0:
            self.silent_steps = 0
        else:
            self.silent_steps += 1
        self.step_index += 1

    def run_auto(self, steps=256, step_seconds=0.08):
        for step in range(steps):
            input_value = self._next_auto_input()
            self.update(input_value)
            if step % 32 == 0:
                print(f"[auto] step={step:03d} input={input_value:.3f}")
            time.sleep(step_seconds)

    def close(self):
        self.all_notes_off()
        close_port = getattr(self.midi_out, "closePort", None)
        close_port_snake = getattr(self.midi_out, "close_port", None)
        close_method = getattr(self.midi_out, "close", None)

        if callable(close_port):
            close_port()
        elif callable(close_port_snake):
            close_port_snake()
        elif callable(close_method):
            close_method()
        else:
            print("Warning: MIDI output object has no close method.")
        self.lock = None

class AudioInput:
    def __init__(self):
        # Initialize the audio input system.
        self.audio_stream = sd.InputStream(callback=self._audio_callback)
        self.audio_stream.start()
        self.latest_audio_input = None

    def _audio_callback(self, indata, frames, time, status):
        # Process the incoming audio data.
        if status:
            print(f"Audio input status: {status}")
        # Here you could store the audio data for later retrieval in get_input.
        self.latest_audio_input = indata.copy()

    def get_input_fft_16bands(self):
        # Return the next audio input fft value.
        if hasattr(self, "latest_audio_input") and self.latest_audio_input is not None:
            # Convert the FFT result to a list of magnitudes in 16 bands.
            fft_result = np.fft.rfft(self.latest_audio_input, axis=0)
            bands16 = [np.mean(np.abs(fft_result[i::16])) for i in range(16)]
            bands16 = np.array(bands16)  # Convert to numpy array for consistency
            #use only the magnitude of the FFT result for each band
            bands16 = np.abs(bands16)
            return bands16
        
        return 0.0

    def start(self):
        if hasattr(self, "audio_input") and self.audio_input is not None:
            self.audio_input.audio_stream.start()

if __name__ == "__main__":
    print("Tip: set LIFNN_MIDI_PORT to part of the desired output name, e.g. LIFNN_MIDI_PORT=IAC")
    lifnn = LIFNN()
    print("Commands: blank=auto, auto N, density X, chaos X, knobs D C, scale NAME [ROOT], scales, show, q")
    lifnn.set_knobs(lifnn.density, lifnn.chaos)
    lifnn.set_scale(lifnn.scale_name, lifnn.root_note)
    try:
        while True:
            raw_value = input("Enter input or command: ").strip()
            if raw_value.lower() in {"q", "quit", "exit"}:
                break
            if raw_value.lower() == "scales":
                print("Available scales:", ", ".join(sorted(SCALES.keys())))
                continue
            if raw_value.lower() == "show":
                lifnn.set_knobs(lifnn.density, lifnn.chaos)
                lifnn.set_scale(lifnn.scale_name, lifnn.root_note)
                continue
            if raw_value == "":
                lifnn.run_auto()
                continue
            if raw_value.lower().startswith("auto"):
                parts = raw_value.split()
                steps = 256
                if len(parts) > 1:
                    steps = int(parts[1])
                lifnn.run_auto(steps=steps)
                continue
            if raw_value.lower().startswith("density"):
                parts = raw_value.split()
                if len(parts) != 2:
                    print("Usage: density 0.0..1.0")
                    continue
                lifnn.set_knobs(density=float(parts[1]))
                continue
            if raw_value.lower().startswith("chaos"):
                parts = raw_value.split()
                if len(parts) != 2:
                    print("Usage: chaos 0.0..1.0")
                    continue
                lifnn.set_knobs(chaos=float(parts[1]))
                continue
            if raw_value.lower().startswith("knobs"):
                parts = raw_value.split()
                if len(parts) != 3:
                    print("Usage: knobs <density 0..1> <chaos 0..1>")
                    continue
                lifnn.set_knobs(density=float(parts[1]), chaos=float(parts[2]))
                continue
            if raw_value.lower().startswith("scale"):
                parts = raw_value.split()
                if len(parts) not in {2, 3}:
                    print("Usage: scale <name> [root_midi_note 0..127]")
                    continue
                root = lifnn.root_note if len(parts) == 2 else int(parts[2])
                lifnn.set_scale(scale_name=parts[1], root_note=root)
                continue
            input_value = float(raw_value)
            lifnn.update(input_value)
    except ValueError:
        print("Invalid number input. Exiting.")
    except KeyboardInterrupt:
        pass
    finally:
        lifnn.close()