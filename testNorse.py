import norse

# Test Norse library import
print("Norse library imported successfully")

# Test Norse LIF neuron model
import torch
from norse.torch import LIFCell

# Create a LIF neuron
lif = LIFCell()

# Create input tensor
input = torch.zeros(1, 1)

# Forward pass through LIF neuron
output, state = lif(input)
print("LIF neuron output:", output)
print("LIF neuron state:", state)

# Test Norse LIF neuron with multiple time steps
time_steps = 5
input_sequence = torch.zeros(time_steps, 1, 1)
state = None
for t in range(time_steps):
    output, state = lif(input_sequence[t], state)
    print(f"Time step {t} output:", output)
    print(f"Time step {t} state:", state)

# Test Norse LIF neuron with random input sequence
input_sequence_random = torch.rand(time_steps, 1, 1)
state = None
for t in range(time_steps):
    output, state = lif(input_sequence_random[t], state)
    print(f"Random input time step {t} output:", output)
    print(f"Random input time step {t} state:", state)

# small fm synth patch implementation
import numpy as np
import sounddevice as sd
class FMSynth:
    def __init__(self, carrier_freq=440.0, modulator_freq=220.0, modulation_index=2.0, fs=44100, duration=1.0):
        self.carrier_freq = carrier_freq
        self.modulator_freq = modulator_freq
        self.modulation_index = modulation_index
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    def generate(self):
        modulator = np.sin(2 * np.pi * self.modulator_freq * self.t)
        carrier = np.sin(2 * np.pi * self.carrier_freq * self.t + self.modulation_index * modulator)
        return carrier

    def play(self):
        signal = self.generate()
        sd.play(signal, self.fs)
        sd.wait()

# build a 16 neuron LIF network that triggers the FM synth when a neuron spikes
import torch
from norse.torch import LIFCell

class LIFNetwork:
    def __init__(self, num_neurons=16):
        self.num_neurons = num_neurons
        self.neurons = [LIFCell() for _ in range(num_neurons)]
        self.synths = [FMSynth(carrier_freq=440.0 + i * 20) for i in range(num_neurons)]

    def forward(self, input_sequence):
        states = [None] * self.num_neurons
        for t in range(input_sequence.size(0)):
            for i, neuron in enumerate(self.neurons):
                output, states[i] = neuron(input_sequence[t, i].unsqueeze(0).unsqueeze(0), states[i])
                if output.item() > 0:  # neuron spikes
                    self.synths[i].play()


# Example usage
if __name__ == "__main__":
    network = LIFNetwork()
    input_sequence = torch.rand(100, 16)  # 100 time steps, 16 neurons
    network.forward(input_sequence)
