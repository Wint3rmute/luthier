import math
from enum import Enum
import subprocess
import time
from abc import ABC, abstractmethod, abstractstaticmethod
from dataclasses import dataclass, is_dataclass
from functools import cache
from itertools import count
from typing import Optional

SAMPLE_RATE = 48000

NodeId = int


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
        return sorted(
            field for field in dir(self.__class__.Outputs) if not field.startswith("_")
        )

    @cache
    def input_names(self) -> list[str]:
        return sorted(
            field for field in dir(self.__class__.Inputs) if not field.startswith("_")
        )

    def get_output_by_index(self, output_index: int) -> float:
        return float(getattr(self.outputs, self.output_names()[output_index]))

    def set_input_by_index(self, input_index: int, input_value: float) -> None:
        setattr(self.inputs, self.input_names()[input_index], input_value)

    @abstractmethod
    def tick(self) -> None:
        ...


class DspGraph:
    def __init__(self) -> None:
        self.nodes: dict[NodeId, DspNode] = {}
        self.connections: list[DspConnection] = []
        self.node_id_counter = count()

    def add_node(self, node: DspNode) -> int:
        """
        Adds a node to the graph and return its unique ID within the graph
        """
        node.node_id = self._get_next_node_id()
        self.nodes[node.node_id] = node
        return node.node_id

    def _get_next_node_id(self) -> NodeId:
        return next(self.node_id_counter)

    def tick(self) -> None:
        for connection in self.connections:
            output_node = self.nodes[connection.from_node]
            value_on_output = output_node.get_output_by_index(connection.from_output)

            input_node = self.nodes[connection.to_node]
            input_node.set_input_by_index(connection.to_input, value_on_output)

        for node in self.nodes.values():
            node.tick()

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

        self.connections.append(
            DspConnection(
                from_node_index,
                from_node.output_names().index(from_output),
                to_node_index,
                to_node.input_names().index(to_input),
            )
        )

    def draw(self) -> bytes:
        result = """
digraph g {
splines="polyline"
fontname="Helvetica,Arial,sans-serif"
node [fontname="Helvetica,Arial,sans-serif"]
edge [fontname="Helvetica,Arial,sans-serif"]
graph [
rankdir = "LR"
];
node [
fontsize = "16"
shape = "record"
];
edge [
];
        """

        for node in self.nodes.values():
            result += f"""
"node{node.node_id}" [
label = "<f0>{node.__class__.__name__} """

            if isinstance(node, Param):
                result += f"| ⇒ {node.outputs.output:.2E}"

            for input in node.input_names():
                result += f"|<{input}> ○ {input}  "

            for output in node.output_names():
                result += f"|<{output}> {output} ●"
            result += '"\n];'

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

        # print(result)
        graphviz_dot_process = subprocess.Popen(
            ["dot", "-T", "jpg"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

        out, errors = graphviz_dot_process.communicate(result.encode())

        if errors != None:
            raise ValueError(f"Errors while running dot: {errors!r}")

        return out


class AdsrPhase(Enum):
    ATTACK = 0
    DECAY = 1
    SUSTAIN = 2
    RELEASE = 3
    FINISHED = 4

class ADSR(DspNode):
    @dataclass
    class Inputs:
        input: float = 0
        attack: float = 0
        decay: float = 0
        # sustain: float = 0
        # release: float = 0

    @dataclass
    class Outputs:
        output: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.phase = AdsrPhase.ATTACK  # Controls the envelope logic
        self.state = 0.0  # Value by which the input will be multiplied

    def tick(self) -> None:
        if self.phase == AdsrPhase.ATTACK:
            self.state += self.inputs.attack
            if self.state > 1.0:
                self.state = 1.0
                self.phase = AdsrPhase.DECAY
        elif self.phase == AdsrPhase.DECAY:
            self.state -= self.inputs.decay
            if self.state < 0.0:
                self.state = 0.0

        self.outputs.output = self.inputs.input * self.state



class Sum(DspNode):
    @dataclass
    class Inputs:
        in_1: float = 0
        in_2: float = 0

    @dataclass
    class Outputs:
        out: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.inputs = self.Inputs
        self.outputs = self.Outputs

    def tick(self) -> None:
        self.outputs.out = self.inputs.in_1 + self.inputs.in_2


class Output(DspNode):
    @dataclass
    class Inputs:
        input: float = 0

    @dataclass
    class Outputs:
        """Reading from an output node is handled by DspGraph's logic"""

    def tick(self) -> None:
        pass


class SineOscilator(DspNode):
    @dataclass
    class Inputs:
        frequency: float = 0
        modulation: float = 0

    @dataclass
    class Outputs:
        output: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.inputs: SineOscilator.Inputs  # type: ignore
        self.outputs: SineOscilator.Outputs  # type: ignore
        self.phase = 0.0

    def tick(self) -> None:
        self.phase_diff = (2.0 * math.pi * self.inputs.frequency) / SAMPLE_RATE
        self.outputs.output = math.sin(self.phase + self.inputs.modulation)
        self.phase += self.phase_diff


class Param(DspNode):
    @dataclass
    class Inputs:
        ...

    @dataclass
    class Outputs:
        output: float = 0

    def __init__(self) -> None:
        super().__init__()
        self.outputs: Param.Outputs  # type: ignore
        self.inputs: Param.Inputs  # type: ignore

    def set_value(self, value: float) -> None: 
        self.outputs.output = value

    def tick(self) -> None:
        pass


# class Multiplier(DspNode):
#     @staticmethod
#     def inputs() -> list[str]:
#         return ["input", "scale"]

#     @staticmethod
#     def outputs() -> list[str]:
#         return ["output"]

#     def tick(self):
#         pass


if __name__ == "__main__":
    g = DspGraph()

    sine_1 = g.add_node(SineOscilator())
    sine_2 = g.add_node(SineOscilator())
    adsr = g.add_node(ADSR())
    sum = g.add_node(Sum())
    output = g.add_node(Output())
    output = g.add_node(Output())
    source_1 = g.add_node(Param())
    source_2 = g.add_node(Param())

    g.nodes[source_2].set_value(440 * 1)

    g.nodes[adsr].inputs.attack = 1.0
    g.nodes[adsr].inputs.decay = 0.0001

    g.patch(sine_1, "output", sine_2, "modulation")
    g.patch(sine_2, "output", output, "input")
    g.patch(source_1, "output", sine_2, "frequency")
    g.patch(source_2, "output", sine_1, "frequency")

    image = g.draw()
    with open("/tmp/image.png", "wb") as image_file:
        image_file.write(image)

    # for i in range(100):
    # while True:
    #     g.tick()
    #     # print(g.nodes[sine_2].outputs.output)
    #     print(g.nodes[output].inputs.input)
    #     time.sleep(0.01)
    #     # break
