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
from norse.torch.functional.lif import LIFParameters

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


#the following code is for the RhythmicStructure class

# Polyrhythm matrix defining various meters, their tempo, and the length and weight of each beat within the meter.
# Tempo is to be induced from audio input so that meters with same tempo last the same time and meters with different tempos shift in phase or form integer relationships
POLYRHYTHM_MATRIX = [
    {"meter": "8/8", "tempo": None, 1: {"length": 1, "weight": 1}, 2: {"length": 1, "weight": 1/16}, 3: {"length": 1, "weight": 1/8}, 4: {"length": 1, "weight": 1/16}, 5: {"length": 1, "weight": 1}, 6: {"length": 1, "weight": 1/16}, 7: {"length": 1, "weight": 1/8}, 8: {"length": 1, "weight": 1}, 9: {"length": 1, "weight": 1/16}, 10: {"length": 1, "weight": 1/8}, 11: {"length": 1, "weight": 1/16}, 12: {"length": 1, "weight": 1}, 13: {"length": 1, "weight": 1}, 14: {"length": 1, "weight": 1/16}, 15: {"length": 1, "weight": 1/8}, 16: {"length": 1, "weight": 1/16}},
    {"meter" : "15/8", "tempo": None, 1: {"length": 16/15, "weight": 1}, 2: {"length": 16/15, "weight": 1/16}, 3: {"length": 16/15, "weight": 1/8}, 4: {"length": 16/15, "weight": 1/16}, 5: {"length": 16/15, "weight": 1}, 6: {"length": 16/15, "weight": 1/16}, 7: {"length": 16/15, "weight": 1/8}, 8: {"length": 16/15, "weight": 1}, 9: {"length": 16/15, "weight": 1/16}, 10: {"length": 16/15, "weight": 1/8}, 11: {"length": 16/15, "weight": 1/16}, 12: {"length": 16/15, "weight": 1}, 13: {"length": 16/15, "weight": 1}, 14: {"length": 16/15, "weight": 1/16}, 15: {"length": 16/15, "weight": 1/8}},
    {"meter" : "7/4", "tempo": None,  1: {"length": 16/14, "weight": 1}, 2: {"length": 16/14, "weight": 1/16}, 3: {"length": 16/14, "weight": 1/8}, 4: {"length": 16/14, "weight": 1/16}, 5: {"length": 16/14, "weight": 1}, 6: {"length": 16/14, "weight": 1/16}, 7: {"length": 16/14, "weight": 1/8}, 8: {"length": 16/14, "weight": 1}, 9: {"length": 16/14, "weight": 1/16}, 10: {"length": 16/14, "weight": 1/8}, 11: {"length": 16/14, "weight": 1/16}, 12: {"length": 16/14, "weight": 1}, 13: {"length": 16/14, "weight": 1}, 14: {"length": 16/14, "weight": 1/16 }},
    {"meter" : "13/8", "tempo": None, 1: {"length": 16/13, "weight": 1}, 2: {"length": 16/13, "weight": 1/16}, 3: {"length": 16/13, "weight": 1/8}, 4: {"length": 16/13, "weight": 1/16}, 5: {"length": 16/13, "weight": 1}, 6: {"length": 16/13, "weight": 1/16}, 7: {"length": 16/13, "weight": 1/8}, 8: {"length": 16/13, "weight": 1}, 9: {"length": 16/13, "weight": 1/16}, 10: {"length": 16/13, "weight": 1/8}, 11: {"length": 16/13, "weight": 1/16}, 12: {"length": 16/13, "weight": 1}, 13: {"length": 16/13, "weight": 1}},
    {"meter" : "12/8", "tempo": None, 1: {"length": 16/12, "weight": 1}, 2: {"length": 16/12, "weight": 1/16}, 3: {"length": 16/12, "weight": 1/8}, 4: {"length": 16/12, "weight": 1/16}, 5: {"length": 16/12, "weight": 1}, 6: {"length": 16/12, "weight": 1/16}, 7: {"length": 16/12, "weight": 1/8}, 8: {"length": 16/12, "weight": 1}, 9: {"length": 16/12, "weight": 1/16}, 10: {"length": 16/12, "weight": 1/8}, 11: {"length": 16/12, "weight": 1/16}, 12: {"length": 16/12, "weight": 1}},
    {"meter" : "11/8", "tempo": None, 1: {"length": 16/11, "weight": 1}, 2: {"length": 16/11, "weight": 1/16}, 3: {"length": 16/11, "weight": 1/8}, 4: {"length": 16/11, "weight": 1/16}, 5: {"length": 16/11, "weight": 1}, 6: {"length": 16/11, "weight": 1/16}, 7: {"length": 16/11, "weight": 1/8}, 8: {"length": 16/11, "weight": 1}, 9: {"length": 16/11, "weight": 1/16}, 10: {"length": 16/11, "weight": 1/8}, 11: {"length": 16/11, "weight": 1}},
    {"meter" : "10/8", "tempo": None, 1: {"length": 16/10, "weight": 1}, 2: {"length": 16/10, "weight": 1/16}, 3: {"length": 16/10, "weight": 1/8}, 4: {"length": 16/10, "weight": 1/16}, 5: {"length": 16/10, "weight": 1}, 6: {"length": 16/10, "weight": 1/16}, 7: {"length": 16/10, "weight": 1/8}, 8: {"length": 16/10, "weight": 1}, 9: {"length": 16/10, "weight": 1/16}, 10: {"length": 16/10, "weight": 1}},
    {"meter" : "9/8", "tempo": None, 1: {"length": 16/9, "weight": 1}, 2: {"length": 16/9, "weight": 1/16}, 3: {"length": 16/9, "weight": 1/8}, 4: {"length": 16/9, "weight": 1/16}, 5: {"length": 16/9, "weight": 1}, 6: {"length": 16/9, "weight": 1/16}, 7: {"length": 16/9, "weight": 1/8}, 8: {"length": 16/9, "weight": 1}, 9: {"length": 16/9, "weight": 1/16}},
    {"meter" : "4/4", "tempo": None, 1: {"length": 16/8, "weight": 1}, 2: {"length": 16/8, "weight": 1/16}, 3: {"length": 16/8, "weight": 1/8}, 4: {"length": 16/8, "weight": 1/16}, 5: {"length": 16/8, "weight": 1}, 6: {"length": 16/8, "weight": 1/16}, 7: {"length": 16/8, "weight": 1/8}, 8: {"length": 16/8, "weight": 1}},
    {"meter" : "7/8", "tempo": None, 1: {"length": 16/7, "weight": 1}, 2: {"length": 16/7, "weight": 1/16}, 3: {"length": 16/7, "weight": 1/8}, 4: {"length": 16/7, "weight": 1/16}, 5: {"length": 16/7, "weight": 1}, 6: {"length": 16/7, "weight": 1/16}, 7: {"length": 16/7, "weight": 1/8}},
    {"meter" : "6/8", "tempo": None, 1: {"length": 16/6, "weight": 1}, 2: {"length": 16/6, "weight": 1/16}, 3: {"length": 16/6, "weight": 1/8}, 4: {"length": 16/6, "weight": 1/16}, 5: {"length": 16/6, "weight": 1}, 6: {"length": 16/6, "weight": 1/16}},
    {"meter" : "5/8", "tempo": None, 1: {"length": 16/5, "weight": 1}, 2: {"length": 16/5, "weight": 1/16}, 3: {"length": 16/5, "weight": 1/8}, 4: {"length": 16/5, "weight": 1/16}, 5: {"length": 16/5, "weight": 1}},
    {"meter" : "2/4", "tempo": None, 1: {"length": 16/4, "weight": 1}, 2: {"length": 16/4, "weight": 1/16}, 3: {"length": 16/4, "weight": 1/8}, 4: {"length": 16/4, "weight": 1/16}},
    {"meter" : "3/8", "tempo": None, 1: {"length": 16/3, "weight": 1}, 2: {"length": 16/3, "weight": 1/16}, 3: {"length": 16/3, "weight": 1/8}},
    {"meter" : "2/4", "tempo": None, 1: {"length": 16/2, "weight": 1}, 2: {"length": 16/2, "weight": 1/16}},
    {"meter" : "1/1", "tempo": None, 1: {"length": 16, "weight": 1}}
]

class PolyRhythm:
    def __init__(self, meters):
        self.meters = meters # this holds one or more meters in the poly-rhythm structure
        self.base_subdivision = 16 #base subdivision for the poly-rhythm cells
        self.dt = None #time step for the poly-rhythm cells, to be set once tempo is known
        self.seconds_per_beat = None
        self.base_unit_seconds = None
        self.target_period_seconds = None
        self.weight = None

        self.lif_cells = [] # list to hold LIFCellMeterWrapper instances

    def add_lif_cell(self, lif_cell_wrapper):
        self.lif_cells.append(lif_cell_wrapper)
        lif_cell_wrapper.poly_rhythm = self
        return lif_cell_wrapper

    def set_tempo(self, tempo):
        self.dt = 60 / (tempo * self.base_subdivision)
        self.seconds_per_beat = 60 / tempo
        self.base_unit_seconds = self.seconds_per_beat * self.base_subdivision
        for lif_cell_wrapper in self.lif_cells:
            lif_cell_wrapper.dt = self.dt
            lif_cell_wrapper.target_period_seconds = lif_cell_wrapper.meter_params.get("length", 1) * self.dt
            lif_cell_wrapper.weight = lif_cell_wrapper.meter_params.get("weight", 1)
            lif_cell_wrapper.poly_rhythm = self

class LIFCellMeterWrapper:
    # Wrapper class for LIF cells of one particular meter in the poly-rhythm structure
    def __init__(self, lif_cells):
        self.lif_cells = lif_cells
        self.tempo = None
        self._induce_tempo_from_audio(None)
        self.dt = 60 / (self.tempo * 16) if self.tempo else None

    def _induce_tempo_from_audio(self, audio_input):
        if self.tempo is None:
            # Here you would implement logic to induce tempo from the audio input
            # For now, we just set a default tempo
            self.tempo = 60
    
    def create_lif_cells_and_tune(self, params):
        self.states = [cell.initial_state(torch.zeros(1)) for cell in self.lif_cells] 
        self.meter_params = params
        self.dt = 60 / (self.tempo * 16) if self.tempo else None
        self.target_period_seconds = self.meter_params.get("length", 1) * self.dt if self.dt else None
        self.weight = self.meter_params.get("weight", 1)
        p = LIFParameters(tau_syn_inv= torch.as_tensor(1/0.005),
                          tau_mem_inv= torch.as_tensor(1/0.02),
                          v_leak= torch.as_tensor(0.0),
                          v_th= torch.as_tensor(1.0),
                          v_reset= torch.as_tensor(0.0))
        self.lif_cells = LIFCell(p=p, dt=self.dt)


class RhythmicStructure:
    def __init__(self):
        self.cell_amount = len(POLYRHYTHM_MATRIX)
        self.current_pattern = None
        self.spike = False
        self.lif_cell_wrappers = [LIFCellMeterWrapper(LIFCell(p=LIFParameters(tau_syn_inv= torch.as_tensor(1/0.005),
                                                                                  tau_mem_inv= torch.as_tensor(1/0.02),
                                                                                  v_leak= torch.as_tensor(0.0),
                                                                                  v_th= torch.as_tensor(1.0),
                                                                                  v_reset= torch.as_tensor(0.0)), dt=60/(60*16))) for params in self.tuning_parameters]
        self._tune_cells(self.tuning_parameters)

    def _tune_cells(self, tuning_parameters=None):
        if tuning_parameters is not None:
            self.tuning_parameters = tuning_parameters
        for wrapper, params in zip(self.lif_cell_wrappers, self.tuning_parameters):
            wrapper.create_lif_cells_and_tune(params)

    def detect_rhythm(self, new_pattern):
        # patterns are one or more of the rhythmic structures defined above in POLYRHYTHM_MATRIX
        if new_pattern != self.current_pattern:
            self.spike = True
            self.current_pattern = new_pattern
        else:
            self.spike = False
        return self.spike

    def generate_event(self):
        self.spike = any(wrapper.lif_cells.spike for wrapper in self.lif_cell_wrappers if hasattr(wrapper.lif_cells, 'spike'))
        return self.spike

class LowLevelEventGenerator:
    def __init__(self, cell_amount=128):
        self.cell_amount = cell_amount
        self.spike = False
        self.lif_cells = torch.nn.ModuleList([LIFCell(p=LIFParameters(tau_syn_inv= torch.as_tensor(1/0.005),
                                                                     tau_mem_inv= torch.as_tensor(1/0.02),
                                                                     v_leak= torch.as_tensor(0.0),
                                                                     v_th= torch.as_tensor(1.0),
                                                                     v_reset= torch.as_tensor(0.0)), dt=60/(60*16)) for _ in range(cell_amount)])

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