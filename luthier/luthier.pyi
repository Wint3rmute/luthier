import numpy
import numpy.typing

F64NDArray = numpy.typing.NDArray[numpy.float64]

class SineOscillator: ...

class ADSR:
    input_attack: float
    input_sustain: float
    input_decay: float
    input_release: float

class Multiplier:
    input_scale: float
    input_input: float
    output_output: float

class HarmonicMultiplier:
    input_scale: float
    input_input: float
    output_output: float

class DspGraph:
    def play(self, num_samples: int) -> F64NDArray: ...
    def draw(self) -> bytes: ...
    def patch(
        self,
        from_node_id: int,
        from_output_name: str,
        to_node_id: int,
        to_input_name: str,
    ) -> None: ...
    def add_sine(self, sine: SineOscillator) -> int: ...
    def add_adsr(self, adsr: ADSR) -> int: ...
    def add_multiplier(self, multiplier: Multiplier) -> int: ...
    def add_harmonic_multiplier(self, multiplier: HarmonicMultiplier) -> int: ...

    speaker_node_id: int
    amp_adsr_node_id: int
