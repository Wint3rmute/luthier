import math
import matplotlib
import random
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import cache, cached_property
from itertools import count
from typing import Any, Optional

import audioflux as af
import librosa
import matplotlib.pyplot as plt
import numpy
import numpy.typing
from audioflux.display import fill_spec
from audioflux.type import SpectralFilterBankScaleType
from dtw import dtw
from IPython.display import Audio, display

# SAMPLE_RATE = 48000
SAMPLE_RATE = 22050
BASE_FREQUENCY = 0.440  # Maps to C4 #  440 * 0.01

NodeId = int
AudioBuffer = numpy.typing.NDArray[numpy.float64]
MfccArray = numpy.typing.NDArray[numpy.float64]


class DspConnection:
    def __init__(
        self, from_node: NodeId, from_output: int, to_node: NodeId, to_input: int
    ):
        self.from_node = from_node
        self.from_output = from_output

        self.to_node = to_node
        self.to_input = to_input

        self.value = 0.0


class DspNode(ABC):
    class Inputs:
        def __init__(self) -> None:
            raise ValueError("You should override the Inputs class in you DspNode")

    class Outputs:
        def __init__(self) -> None:
            raise ValueError("You should override the Outputs class in you DspNode")

    def __init__(self) -> None:
        """
        Each node has a unique `node_id`, which is assigned by the parent `DspGraph`.
        `node_id` is used to store the connections (patching) with other nodes.
        """
        self.node_id: Optional[NodeId] = None

        if not is_dataclass(self.__class__.Inputs):
            raise ValueError("Inputs must be a dataclass")

        if not is_dataclass(self.__class__.Outputs):
            raise ValueError("Outputs must be a dataclass")

        self.inputs = self.__class__.Inputs()
        self.outputs = self.__class__.Outputs()

    @cache
    def output_names(self) -> list[str]:
        # __import__('pdb').set_trace()
        try:
            return sorted(
                field for field in dir(self.outputs) if not field.startswith("_")
            )
        except ValueError:
            return []

    @cache
    def input_names(self) -> list[str]:
        try:
            return sorted(
                field for field in dir(self.inputs) if not field.startswith("_")
            )
        except ValueError:
            return []

    def get_output_by_index(self, output_index: int) -> float:
        return float(getattr(self.outputs, self.output_names()[output_index]))

    def set_input_by_index(self, input_index: int, input_value: float) -> None:
        setattr(self.inputs, self.input_names()[input_index], input_value)

    @abstractmethod
    def tick(self) -> None:
        ...


class Sample:
    def __init__(self, audio_buffer: AudioBuffer) -> None:
        if sum(abs(audio_buffer)) == 0:
            self.buffer = audio_buffer
        else:
            self.buffer = audio_buffer / numpy.linalg.norm(audio_buffer)
            self.buffer /= max(self.buffer)
        # self.buffer /= max(self.buffer) 

    def __len__(self) -> int:
        """Returns the length of the underlying audio buffer"""
        return len(self.buffer)

    @cached_property
    def mfcc(self) -> MfccArray:
        return librosa.feature.mfcc(y=self.buffer, sr=SAMPLE_RATE)

    @cached_property
    def spectrogram(self) -> Any:
        """Returns objects required to plot a spectrogram in matplotlib"""
        # Create BFT object and extract mel spectrogram
        bft_obj = af.BFT(
            num=128,
            radix2_exp=12,
            samplate=SAMPLE_RATE,
            scale_type=SpectralFilterBankScaleType.MEL,
        )

        spec_arr = bft_obj.bft(self.buffer)
        spec_arr = numpy.abs(spec_arr)
        return bft_obj, spec_arr

    def plot_spectrogram(self, ax: plt.Axes, title: str = "Mel spectrogram") -> None:
        bft_obj, spec_arr = self.spectrogram

        fill_spec(
            spec_arr,
            axes=ax,
            x_coords=bft_obj.x_coords(len(self)),
            y_coords=bft_obj.y_coords(),
            x_axis="time",
            y_axis="log",
            title=title,
        )

    def plot_mfcc(self, ax: plt.Axes, title: str = "MFCC") -> None:
        ax.set_title(title)
        librosa.display.specshow(self.mfcc, ax=ax)

    def plot_waveform(
        self, ax: plt.Axes, num_samples: int = 400, title: str = "Waveform"
    ) -> None:
        ax.set_title(title)
        ax.plot(self.buffer[:num_samples])

    def show_player(self) -> None:
        """Display playble audio widget in Jupyter"""
        display(Audio(data=self.buffer, rate=SAMPLE_RATE))  # type: ignore

    def mfcc_distance(self, other: "Sample") -> float:
        dist, cost, acc_cost, path = dtw(
            self.mfcc.T, other.mfcc.T, dist=lambda x, y: numpy.linalg.norm(x - y, ord=1)
        )
        return float(dist)


class DspGraph:
    def __init__(self) -> None:
        self.nodes: dict[NodeId, DspNode] = {}
        self.connections: list[DspConnection] = []
        self.node_id_counter = count()
        self.speaker = self._add_node_no_check(Speaker())

    def iter_params(self) -> int:
        for node in self.nodes.values():
            if isinstance(node, Param):
                yield node

    def num_params(self) -> int:
        return len(list(self.iter_params()))

    def assign_params(self, values: list[float]) -> None:
        if len(values) != self.num_params():
            raise ValueError(f"Number of values to assign ({len(values)}) different from number of params ({self.num_params()})")

        for param, new_value in zip(self.iter_params(), values):
            param.set_value(new_value)

    def get_param_values(self) -> list[float]:
        return [p.get_value() for p in self.iter_params() ]

    def add_node(self, node: DspNode) -> int:
        """
        Adds a node to the graph and return its unique ID within the graph
        """
        if isinstance(node, Speaker):
            raise ValueError("Speaker node already present in the graph")

        return self._add_node_no_check(node)

    def play(self, num_samples: int) -> Sample:
        audio_buffer = numpy.zeros(num_samples)

        for index in range(len(audio_buffer)):
            audio_buffer[index] = self.tick()

        return Sample(audio_buffer=audio_buffer)

    def _add_node_no_check(self, node: DspNode) -> int:
        node.node_id = self._get_next_node_id()
        self.nodes[node.node_id] = node
        return node.node_id

    def _get_next_node_id(self) -> NodeId:
        return next(self.node_id_counter)

    def tick(self) -> float:
        """
        Runs a single iteration on all nodes
        and returns the value on the speaker node
        """
        for connection in self.connections:
            output_node = self.nodes[connection.from_node]
            value_on_output = output_node.get_output_by_index(connection.from_output)

            input_node = self.nodes[connection.to_node]
            input_node.set_input_by_index(connection.to_input, value_on_output)

        for node in self.nodes.values():
            node.tick()

        return self.nodes[self.speaker].inputs.input  # type: ignore

    def patch(
        self, from_node_index: int, from_output: str, to_node_index: int, to_input: str
    ) -> None:
        if from_node_index >= len(self.nodes):
            raise ValueError("from_node not found")

        if to_node_index >= len(self.nodes):
            raise ValueError("to_node not found")

        from_node = self.nodes[from_node_index]
        to_node = self.nodes[to_node_index]

        if from_output not in from_node.output_names():
            # __import__("pdb").set_trace()
            raise ValueError(f"output {from_output} not found in {from_node}")

        if to_input not in to_node.input_names():
            raise ValueError(f"input {to_input} not found in {to_node}")

        for connection in self.connections:
            if (
                connection.to_node == to_node_index
                and connection.to_input == to_node.input_names().index(to_input)
            ):
                # print(
                #     f"Ignoring patching more than one input to {to_node_index}/{to_input}"
                # )
                return

        self.connections.append(
            DspConnection(
                from_node_index,
                from_node.output_names().index(from_output),
                to_node_index,
                to_node.input_names().index(to_input),
            )
        )

    def save_video(self, output_path: str = "/tmp/output.mp4") -> Sample:
        with tempfile.TemporaryDirectory() as dir:
            samples = numpy.zeros(SAMPLE_RATE)
            for i in range(len(samples)):
                samples[i] = graph.tick()
                if i % 100 == 0:
                    with open(dir + f"/{i}.png", "wb") as drawing_file:
                        drawing_file.write(graph.draw())

            os.system(f"cat {dir}/*.png | ffmpeg -y -f image2pipe -i - {output_path}")

        return Sample(audio_buffer=samples)

    def draw(self) -> bytes:
        cmap = matplotlib.colormaps.get_cmap('bwr')
        result = """
digraph g {
    splines="polyline"
    rankdir = "LR"

    fontname="Fira Code"
    node [fontname="Fira Code"]
"""

        for node in self.nodes.values():
            node_to_color = {
                SineOscillator: "#FF5370",
                Param: "white",
                Doubler: "#C792EA",
                ADSR: "#FFCB6B",
            }

            color = node_to_color.get(node.__class__, "white")

            result += f"""
"node{node.node_id}" [ 
    shape = none
    label = <<table border="0" cellspacing="0">

        <tr><td border="1" bgcolor="{color}">{node.__class__.__name__}</td></tr>
"""
            if isinstance(node, Param):
                param_value = float(node.outputs.output)
                color = matplotlib.colors.to_hex(cmap((param_value + 1) / 2))
                result += f'<tr><td border="1" bgcolor="{color}"> ⇒ {param_value:.3} </td></tr> \n'

            for input in node.input_names():
                param_value = float(getattr(node.inputs, input))
                color = matplotlib.colors.to_hex(cmap((param_value + 1) / 2))
                result += f'<tr><td border="1" port="{input}" bgcolor="{color}"> ○ {input} </td></tr> \n'

            for output in node.output_names():
                result += f'<tr><td border="1" port="{output}"> {output} ● </td></tr> \n'

            result += '</table>>\n];'

        for connection in self.connections:
            output_name = self.nodes[connection.from_node].output_names()[
                connection.from_output
            ]
            input_name = self.nodes[connection.to_node].input_names()[
                connection.to_input
            ]

            result += f"""
"node{connection.from_node}":{output_name} -> "node{connection.to_node}":{input_name} [];
            """

        result += "\n}"
        graphviz_dot_process = subprocess.Popen(
            ["dot", "-T", "png"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

        out, errors = graphviz_dot_process.communicate(result.encode())

        if errors is not None:
            raise ValueError(f"Errors while running dot: {errors!r}")

        return out


class AdsrPhase(Enum):
    ATTACK = 0
    SUSTAIN = 1
    RELEASE = 2


class ADSR(DspNode):
    @dataclass
    class Inputs:
        input: float = 0
        attack: float = 1
        # There's no key input in the simulation, hence sustain is a parameter
        # defining how long the sound stay in higest-level attack velocity
        # before the release phase
        sustain: float = 0.001
        release: float = 0.001

    @dataclass
    class Outputs:
        output: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.inputs: ADSR.Inputs
        self.outputs: ADSR.Outputs

        self.phase = AdsrPhase.ATTACK  # Controls the envelope logic
        self.state = 0.0  # Value by which the input will be multiplied
        self.sustain_state = 0.0

    def tick(self) -> None:
        if self.phase == AdsrPhase.ATTACK:
            self.state += abs(self.inputs.attack)
            if self.state > 1.0:
                self.state = 1.0
                self.phase = AdsrPhase.SUSTAIN

        elif self.phase == AdsrPhase.SUSTAIN:
            self.sustain_state += abs(self.inputs.sustain)
            if self.sustain_state >= 1.0:
                self.phase = AdsrPhase.RELEASE

        elif self.phase == AdsrPhase.RELEASE:
            self.state -= abs(self.inputs.release)
            if self.state < 0.0:
                self.state = 0.0

        self.outputs.output = self.inputs.input * self.state


class Sum(DspNode):
    @dataclass
    class Inputs:
        in_1: float = 0
        in_2: float = 0
        in_3: float = 0
        in_4: float = 0

    @dataclass
    class Outputs:
        out: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.inputs: Sum.Inputs
        self.outputs: Sum.Outputs

    def tick(self) -> None:
        self.outputs.out = (
            self.inputs.in_1 + self.inputs.in_2 + self.inputs.in_3 + self.inputs.in_4
        )


class Speaker(DspNode):
    @dataclass
    class Inputs:
        input: float = 0

    @dataclass
    class Outputs:
        pass
        """Reading from an output node is handled by DspGraph's logic"""

    def tick(self) -> None:
        pass


class SineOscillator(DspNode):
    @dataclass
    class Inputs:
        frequency: float = 0
        modulation: float = 0

    @dataclass
    class Outputs:
        output: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.inputs: SineOscillator.Inputs
        self.outputs: SineOscillator.Outputs
        self.phase = 0.0

    def tick(self) -> None:
        frequency = abs(self.inputs.frequency * 1000)
        self.phase_diff = (2.0 * math.pi * frequency) / SAMPLE_RATE
        self.outputs.output = math.sin(self.phase + self.inputs.modulation)
        self.phase += self.phase_diff

        while self.phase > math.pi * 2.0:
            self.phase -= math.pi * 2.0


class BaseFrequency(DspNode):
    @dataclass
    class Inputs:
        ...

    class Outputs:
        output: float = BASE_FREQUENCY


class Param(DspNode):
    @dataclass
    class Inputs:
        ...

    @dataclass
    class Outputs:
        output: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.outputs: Param.Outputs
        self.inputs: Param.Inputs

    def set_value(self, value: float) -> None:
        self.outputs.output = value

    def get_value(self) -> float:
        return self.outputs.output

    def tick(self) -> None:
        pass


class Doubler(DspNode):
    @dataclass
    class Inputs:
        input: float = 0

    @dataclass
    class Outputs:
        output: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.inputs: Doubler.Inputs
        self.outputs: Doubler.Outputs

    def tick(self) -> None:
        self.outputs.output = self.inputs.input * 2.0


class Multiplier(DspNode):
    @dataclass
    class Inputs:
        input: float = 0
        scale: float = 1.0

    @dataclass
    class Outputs:
        output: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.inputs: Multiplier.Inputs
        self.outputs: Multiplier.Outputs

    def tick(self) -> None:
        self.outputs.output = self.inputs.input * self.inputs.scale


def draw_to_temp_file(graph: DspGraph) -> None:
    image = graph.draw()
    with open("/tmp/image.png", "wb") as image_file:
        image_file.write(image)


POSSIBLE_NODES_TO_ADD: list[type[DspNode]] = [
    SineOscillator,
    SineOscillator,
    SineOscillator,
    SineOscillator,
    Doubler,
    ADSR,
    Sum,
    Param,
]


def add_random_node(graph: DspGraph) -> None:
    node = random.choice(POSSIBLE_NODES_TO_ADD)()
    graph.add_node(node)


def add_random_connection(graph: DspGraph) -> None:
    node_from_id, node_from = random.choice(list(graph.nodes.items()))
    node_to_id, node_to = random.choice(list(graph.nodes.items()))

    if node_from is node_to:
        return

    if not any(node_from.output_names()):
        return

    if not any(node_to.input_names()):
        return

    output_name = random.choice(node_from.output_names())
    input_name = random.choice(node_to.input_names())

    graph.patch(node_from_id, output_name, node_to_id, input_name)


def remove_random_connection(graph: DspGraph) -> None:
    if not any(graph.connections):
        return

    graph.connections.pop(random.randrange(len(graph.connections)))


def _get_random_param(graph: DspGraph) -> Optional[Param]:
    params = [node for node in graph.nodes.values() if isinstance(node, Param)]
    if any(params):
        result_param = random.choice(params)
        # if result_param.get_value() == BASE_FREQUENCY:
        #     return None
        return result_param

    print("No params found")
    return None

def set_random_param_to_base_frequency(graph: DspGraph) -> None:
    if param := _get_random_param(graph):
        param.set_value(BASE_FREQUENCY)

def set_random_param_to_one(graph: DspGraph) -> None:
    if param := _get_random_param(graph):
        param.set_value(1.0)

def randomize_random_param(graph: DspGraph) -> None:
    if param := _get_random_param(graph):
        param.set_value(random.uniform(-1, 1))


def nudge_random_param(graph: DspGraph) -> None:
    if param := _get_random_param(graph):
        param.set_value(param.get_value() + random.uniform(-0.01, 0.01))


def multiply_random_param_by_harmonic(graph: DspGraph) -> None:
    if param := _get_random_param(graph):
        value = param.get_value()
        param.set_value(
            value
            * random.choice(
                [0.25, 1 / 3, 4 / 5, 0.5, 0.75, 5 / 4, 2 / 3, 2 / 3, 2.5, 3.5, 3]
            )
        )


def get_starting_graph() -> DspGraph:
    graph = DspGraph()

    sine = graph.add_node(SineOscillator())
    amp_adsr = graph.add_node(ADSR())

    _amp_attack = Param()
    _amp_attack.set_value(0.01)
    amp_attack = graph.add_node(_amp_attack)

    _amp_sustain = Param()
    _amp_sustain.set_value(0.001)
    amp_sustain = graph.add_node(_amp_sustain)

    _amp_release = Param()
    _amp_release.set_value(0.0001)
    amp_release = graph.add_node(_amp_release)

    graph.patch(amp_attack, "output", amp_adsr, "attack")
    graph.patch(amp_sustain, "output", amp_adsr, "sustain")
    graph.patch(amp_release, "output", amp_adsr, "release")

    base_frequency_node = Param()
    base_frequency_node.set_value(BASE_FREQUENCY)
    base_frequency = graph.add_node(base_frequency_node)
    graph.base_frequency_node = base_frequency  # type: ignore
    graph.amp_attack = amp_attack # type: ignore
    graph.amp_sustain = amp_sustain  # type: ignore
    graph.amp_release = amp_release  # type: ignore

    graph.patch(base_frequency, "output", sine, "frequency")
    graph.patch(sine, "output", amp_adsr, "input")
    graph.patch(amp_adsr, "output", graph.speaker, "input")

    return graph


if __name__ == "__main__":
    import time
    import os
    import tempfile

    graph = get_starting_graph()
    # a = graph.play(SAMPLE_RATE * 1)

    # __import__("pdb").set_trace()

    for _ in range(20):
        random.choice(
            [
                add_random_node,
                add_random_node,
                add_random_node,
                add_random_node,
                add_random_node,
                add_random_node,
                add_random_node,
                add_random_node,
                add_random_node,
                add_random_node,
                add_random_connection,
                add_random_connection,
                add_random_connection,
                add_random_connection,
                add_random_connection,
                add_random_connection,
                remove_random_connection,
                randomize_random_param,
                randomize_random_param,
                nudge_random_param
            ]
        )(graph)
        draw_to_temp_file(graph)
        # time.sleep(0.1)

    print(graph.num_params())
    print(graph.get_param_values())


    # graph.save_video()

    # for _ in range(10):
    #     draw_to_temp_file(graph)
    #     time.sleep(.5)

    # source_1 = g.add_node(Param())
    # source_2 = g.add_node(Param())

    # g.nodes[source_2].set_value(440 * 1)

    # g.patch(sine_1, "output", sine_2, "modulation")
    # g.patch(sine_2, "output", output, "input")
    # g.patch(source_1, "output", sine_2, "frequency")
    # g.patch(source_2, "output", sine_1, "frequency")

