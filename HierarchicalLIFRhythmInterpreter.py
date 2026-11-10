# three layers: musical form, rhythmic structure, and low-level event generation
# MusicalForm is the highest layer, and moves in time at a relatively slow pace, defining the overall structure of the piece.
# MusicalForm spikes when a significant structural change in the music is detected, such as the start of a new section or phrase.
# RhythmicStructure is the middle layer, and operates at a faster pace than MusicalForm, capturing the underlying rhythmic patterns and meter.
# RhythmicStructure spikes when a notable rhythmic event occurs, such as a strong beat or a syncopated accent.
# RhythmicStructure operates on a faster timescale than MusicalForm, allowing it to respond to more immediate rhythmic cues.
# RhythmicStructure is to be desingned in a way that allows the rhythms to have 3:2 cross-rhythms, enabling polyrhythmic interactions with the underlying meter.
# Low-level event generation is the fastest layer, responsible for producing the actual musical events, such as note on/off messages or control changes.
# It responds to the spikes from both MusicalForm and RhythmicStructure, translating the higher-level structural and rhythmic information into concrete musical actions.
import torch
from norse.torch import LIFCell

class MusicalForm:
    def __init__(self, cell_amount=1024):
        self.cell_amount = cell_amount
        self.current_state = None
        self.spike = False
        self.lif_cells = torch.nn.ModuleList([LIFCell() for _ in range(cell_amount)])

    def detect_change(self, new_state):
        if new_state != self.current_state:
            self.spike = True
            self.current_state = new_state
        else:
            self.spike = False
        return self.spike

POLYRHYTHM_MATRIX = [
    {"meter": "16/8", 1: {"length": 1, "weight": 1}, 2: {"length": 1, "weight": 1/16}, 3: {"length": 1, "weight": 1/8}, 4: {"length": 1, "weight": 1/16}, 5: {"length": 1, "weight": 1}, 6: {"length": 1, "weight": 1/16}, 7: {"length": 1, "weight": 1/8}, 8: {"length": 1, "weight": 1}, 9: {"length": 1, "weight": 1/16}, 10: {"length": 1, "weight": 1/8}, 11: {"length": 1, "weight": 1/16}, 12: {"length": 1, "weight": 1}, 13: {"length": 1, "weight": 1}, 14: {"length": 1, "weight": 1/16}, 15: {"length": 1, "weight": 1/8}, 16: {"length": 1, "weight": 1/16}},
    {"meter" : "15/8", 1: {"length": 16/15, "weight": 1}, 2: {"length": 16/15, "weight": 1/16}, 3: {"length": 16/15, "weight": 1/8}, 4: {"length": 16/15, "weight": 1/16}, 5: {"length": 16/15, "weight": 1}, 6: {"length": 16/15, "weight": 1/16}, 7: {"length": 16/15, "weight": 1/8}, 8: {"length": 16/15, "weight": 1}, 9: {"length": 16/15, "weight": 1/16}, 10: {"length": 16/15, "weight": 1/8}, 11: {"length": 16/15, "weight": 1/16}, 12: {"length": 16/15, "weight": 1}, 13: {"length": 16/15, "weight": 1}, 14: {"length": 16/15, "weight": 1/16}, 15: {"length": 16/15, "weight": 1/8}},
    {"meter" : "14/8", 1: {"length": 16/14, "weight": 1}, 2: {"length": 16/14, "weight": 1/16}, 3: {"length": 16/14, "weight": 1/8}, 4: {"length": 16/14, "weight": 1/16}, 5: {"length": 16/14, "weight": 1}, 6: {"length": 16/14, "weight": 1/16}, 7: {"length": 16/14, "weight": 1/8}, 8: {"length": 16/14, "weight": 1}, 9: {"length": 16/14, "weight": 1/16}, 10: {"length": 16/14, "weight": 1/8}, 11: {"length": 16/14, "weight": 1/16}, 12: {"length": 16/14, "weight": 1}, 13: {"length": 16/14, "weight": 1}, 14: {"length": 16/14, "weight": 1/16 }},
    {"meter" : "13/8", 1: {"length": 16/13, "weight": 1}, 2: {"length": 16/13, "weight": 1/16}, 3: {"length": 16/13, "weight": 1/8}, 4: {"length": 16/13, "weight": 1/16}, 5: {"length": 16/13, "weight": 1}, 6: {"length": 16/13, "weight": 1/16}, 7: {"length": 16/13, "weight": 1/8}, 8: {"length": 16/13, "weight": 1}, 9: {"length": 16/13, "weight": 1/16}, 10: {"length": 16/13, "weight": 1/8}, 11: {"length": 16/13, "weight": 1/16}, 12: {"length": 16/13, "weight": 1}, 13: {"length": 16/13, "weight": 1}},
    {"meter" : "12/8", 1: {"length": 16/12, "weight": 1}, 2: {"length": 16/12, "weight": 1/16}, 3: {"length": 16/12, "weight": 1/8}, 4: {"length": 16/12, "weight": 1/16}, 5: {"length": 16/12, "weight": 1}, 6: {"length": 16/12, "weight": 1/16}, 7: {"length": 16/12, "weight": 1/8}, 8: {"length": 16/12, "weight": 1}, 9: {"length": 16/12, "weight": 1/16}, 10: {"length": 16/12, "weight": 1/8}, 11: {"length": 16/12, "weight": 1/16}, 12: {"length": 16/12, "weight": 1}},
    {"meter" : "11/8", 1: {"length": 16/11, "weight": 1}, 2: {"length": 16/11, "weight": 1/16}, 3: {"length": 16/11, "weight": 1/8}, 4: {"length": 16/11, "weight": 1/16}, 5: {"length": 16/11, "weight": 1}, 6: {"length": 16/11, "weight": 1/16}, 7: {"length": 16/11, "weight": 1/8}, 8: {"length": 16/11, "weight": 1}, 9: {"length": 16/11, "weight": 1/16}, 10: {"length": 16/11, "weight": 1/8}, 11: {"length": 16/11, "weight": 1}},
    {"meter" : "10/8", 1: {"length": 16/10, "weight": 1}, 2: {"length": 16/10, "weight": 1/16}, 3: {"length": 16/10, "weight": 1/8}, 4: {"length": 16/10, "weight": 1/16}, 5: {"length": 16/10, "weight": 1}, 6: {"length": 16/10, "weight": 1/16}, 7: {"length": 16/10, "weight": 1/8}, 8: {"length": 16/10, "weight": 1}, 9: {"length": 16/10, "weight": 1/16}, 10: {"length": 16/10, "weight": 1}},
    {"meter" : "9/8", 1: {"length": 16/9, "weight": 1}, 2: {"length": 16/9, "weight": 1/16}, 3: {"length": 16/9, "weight": 1/8}, 4: {"length": 16/9, "weight": 1/16}, 5: {"length": 16/9, "weight": 1}, 6: {"length": 16/9, "weight": 1/16}, 7: {"length": 16/9, "weight": 1/8}, 8: {"length": 16/9, "weight": 1}, 9: {"length": 16/9, "weight": 1/16}},
    {"meter" : "8/8", 1: {"length": 16/8, "weight": 1}, 2: {"length": 16/8, "weight": 1/16}, 3: {"length": 16/8, "weight": 1/8}, 4: {"length": 16/8, "weight": 1/16}, 5: {"length": 16/8, "weight": 1}, 6: {"length": 16/8, "weight": 1/16}, 7: {"length": 16/8, "weight": 1/8}, 8: {"length": 16/8, "weight": 1}},
    {"meter" : "7/8", 1: {"length": 16/7, "weight": 1}, 2: {"length": 16/7, "weight": 1/16}, 3: {"length": 16/7, "weight": 1/8}, 4: {"length": 16/7, "weight": 1/16}, 5: {"length": 16/7, "weight": 1}, 6: {"length": 16/7, "weight": 1/16}, 7: {"length": 16/7, "weight": 1/8}},
    {"meter" : "6/8", 1: {"length": 16/6, "weight": 1}, 2: {"length": 16/6, "weight": 1/16}, 3: {"length": 16/6, "weight": 1/8}, 4: {"length": 16/6, "weight": 1/16}, 5: {"length": 16/6, "weight": 1}, 6: {"length": 16/6, "weight": 1/16}},
    {"meter" : "5/8", 1: {"length": 16/5, "weight": 1}, 2: {"length": 16/5, "weight": 1/16}, 3: {"length": 16/5, "weight": 1/8}, 4: {"length": 16/5, "weight": 1/16}, 5: {"length": 16/5, "weight": 1}},
    {"meter" : "4/8", 1: {"length": 16/4, "weight": 1}, 2: {"length": 16/4, "weight": 1/16}, 3: {"length": 16/4, "weight": 1/8}, 4: {"length": 16/4, "weight": 1/16}},
    {"meter" : "3/8", 1: {"length": 16/3, "weight": 1}, 2: {"length": 16/3, "weight": 1/16}, 3: {"length": 16/3, "weight": 1/8}},
    {"meter" : "2/8", 1: {"length": 16/2, "weight": 1}, 2: {"length": 16/2, "weight": 1/16}},
    {"meter" : "1/8", 1: {"length": 16/1, "weight": 1}}
]

class LIFCellPolyRhythmWrapper:
    #each LIF cell is wrapped with a poly-rhythm wrapper to allow tuning according to poly-rhythm parameters
    def __init__(self, lif_cell, poly_rhythm_params):
        self.lif_cell = lif_cell
        self.poly_rhythm_params = poly_rhythm_params

    def tune(self, params):
        self.poly_rhythm_params = params
        # tuning is supposed to make the cell respond according to the poly-rhythm parameters
        # Here you would implement the logic to adjust the LIF cell's behavior based on the poly-rhythm parameters
        # For example, you might modify the cell's time constants, thresholds, or other properties
        pass

class RhythmicStructure:
    def __init__(self, cell_amount=136): # default to 136 cells to match the maximum number of rhythmic structures in POLYRHYTHM_MATRIX
        self.cell_amount = cell_amount
        self.current_pattern = None
        self.spike = False
        self.lif_cells = torch.nn.ModuleList([LIFCell() for _ in range(cell_amount)])
        self.tuning_parameters = [POLYRHYTHM_MATRIX[i % len(POLYRHYTHM_MATRIX)] for i in range(cell_amount)]
        self._tune_cells(self.tuning_parameters)

    def _tune_cells(self, tuning_parameters=None):
        if tuning_parameters is not None:
            self.tuning_parameters = tuning_parameters
        for cell, params in zip(self.lif_cells, self.tuning_parameters):
            wrapper = LIFCellPolyRhythmWrapper(cell, params)
            wrapper.tune(params)

    def detect_rhythm(self, new_pattern):
        # patterns are one or more of the rhythmic structures defined above in POLYRHYTHM_MATRIX
        if new_pattern != self.current_pattern:
            self.spike = True
            self.current_pattern = new_pattern
        else:
            self.spike = False
        return self.spike

    def lif_step(self, input):
        output = []
        for cell in self.lif_cells:
            output.append(cell(input))
        return output
    
class LowLevelEventGenerator:
    def __init__(self, cell_amount=64):
        self.cell_amount = cell_amount
        self.spike = False
        self.lif_cells = torch.nn.ModuleList([LIFCell() for _ in range(cell_amount)])

    def generate_event(self, musical_form_spike, rhythmic_structure_spike):
        self.spike = musical_form_spike or rhythmic_structure_spike
        return self.spike

if __name__ == "__main__":
    rhythmic_structure = RhythmicStructure()
    low_level_event_generator = LowLevelEventGenerator()
    # Example usage
    new_pattern = {"meter": "4/8", 1: {"length": 16/4}, 2: {"length": 16/4}, 3: {"length": 16/4}, 4: {"length": 16/4}}
    rhythm_spike = rhythmic_structure.detect_rhythm(new_pattern)
    event_spike = low_level_event_generator.generate_event(False, rhythm_spike)
    print("Rhythm spike:", rhythm_spike)
    print("Event spike:", event_spike)