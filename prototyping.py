import subprocess
from itertools import count
import math
from abc import ABC, abstractmethod, abstractstaticmethod


SAMPLE_RATE = 48000


class DspConnection:
    def __init__(self, from_node: int, from_output: int, to_node: int, to_input: int):
        self.from_node = from_node
        self.from_output = from_output

        self.to_node = to_node
        self.to_input = to_input

        self.value = 0.0


class DspNode(ABC):
    def __init__(self):
        """
        Each node has a unique `node_id`, which is assigned by the parent `DspGraph`.
        `node_id` is used to store the connections (patching) with other nodes.
        """
        self.node_id = None
        # self.node_id = parent_graph.get_next_node_id()
        # parent_graph.nodes.append(self)
        # return self.node_id

    @abstractstaticmethod
    def outputs() -> list[str]:
        ...

    @abstractstaticmethod
    def inputs() -> list[str]:
        ...

    @abstractmethod
    def tick(self, dsp_graph: "DspGraph"):
        ...


class DspGraph:
    def __init__(self):
        self.nodes: list[DspNode] = []
        self.connections: list[DspConnection] = []
        self.node_id_counter = count()

    def add_node(self, node: DspNode) -> int:
        node.node_id = self._get_next_node_id()
        self.nodes.append(node)
        return node.node_id

    def _get_next_node_id(self):
        return next(self.node_id_counter)

    def tick(self):
        for node in self.nodes:
            node.tick(self)

    def patch(
        self, from_node_index: int, from_output: str, to_node_index: int, to_input: str
    ):
        if from_node_index >= len(self.nodes):
            raise ValueError("from_node not found")

        if to_node_index >= len(self.nodes):
            raise ValueError("to_node not found")

        from_node = self.nodes[from_node_index]
        to_node = self.nodes[to_node_index]

        if from_output not in from_node.outputs():
            raise ValueError(f"output {from_output} not found in {from_node}")

        if to_input not in to_node.inputs():
            raise ValueError(f"input {to_input} not found in {to_node}")

        self.connections.append(
            DspConnection(
                from_node_index,
                from_node.outputs().index(from_output),
                to_node_index,
                to_node.inputs().index(to_input),
            )
        )

    def draw(self):
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

        for node in self.nodes:
            result += f"""
"node{node.node_id}" [
label = "<f0>{node.__class__.__name__} """

            for input in node.inputs():
                result += f"|<{input}> ○ {input}  "

            for output in node.outputs():
                result += f"|<{output}> {output} ●"
            result += '"\n];'

        for connection in self.connections:
            output_name = self.nodes[connection.from_node].outputs()[
                connection.from_output
            ]
            input_name = self.nodes[connection.to_node].inputs()[connection.to_input]

            result += f"""
"node{connection.from_node}":{output_name} -> "node{connection.to_node}":{input_name} [];
            """

        result += "\n}"

        print(result)


class ADSR(DspNode):
    @staticmethod
    def inputs() -> list[str]:
        return ["input", "attack", "decay", "sustain", "release"]

    @staticmethod
    def outputs() -> list[str]:
        return ["output"]

    def tick(self):
        pass


class Sum(DspNode):
    @staticmethod
    def inputs() -> list[str]:
        return ["in_1", "in_2"]

    @staticmethod
    def outputs() -> list[str]:
        return ["output"]

    def tick(self):
        pass


class Output(DspNode):
    @staticmethod
    def inputs() -> list[str]:
        return ["input"]

    @staticmethod
    def outputs() -> list[str]:
        return []

    def tick(self):
        pass


class SineOscilator(DspNode):
    @staticmethod
    def inputs() -> list[str]:
        return ["frequency", "modulation"]

    @staticmethod
    def outputs() -> list[str]:
        return ["output"]

    def tick(self):
        return 0.0
        # result = math.sin(self.phase + modulation)
        # self.phase += self.phase_diff
        # return result


class ConstantSource(DspNode):
    @staticmethod
    def inputs() -> list[str]:
        return []

    @staticmethod
    def outputs() -> list[str]:
        return ["output"]

    def tick(self):
        pass


class Multiplier(DspNode):
    @staticmethod
    def inputs() -> list[str]:
        return ["input", "scale"]

    @staticmethod
    def outputs() -> list[str]:
        return ["output"]

    def tick(self):
        pass


if __name__ == "__main__":
    g = DspGraph()

    sine_1 = g.add_node(SineOscilator())
    sine_2 = g.add_node(SineOscilator())
    adsr = g.add_node(ADSR())
    sum = g.add_node(Sum())
    output = g.add_node(Output())

    g.patch(sine_1, "output", sine_2, "modulation")
    g.patch(sine_2, "output", adsr, "input")
    g.patch(sine_1, "output", sum, "in_1")
    g.patch(adsr, "output", sum, "in_2")
    g.patch(sum, "output", output, "input")

    g.draw()
