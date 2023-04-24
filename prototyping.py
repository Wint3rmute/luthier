import subprocess
import math
from abc import ABC, abstractmethod, abstractstaticmethod


SAMPLE_RATE = 48000


class DspConnection:
    def __init__(self, from_node: int, from_output: int, to_node: int, to_input: int):
        self.from_node = from_node
        self.from_output = from_output

        self.to_node = to_node
        self.to_input = to_input


class DspGraph:
    def __init__(self):
        self.nodes: list[DspNode] = []
        self.connections: list[DspConnection] = []

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

        for node_index, node in enumerate(self.nodes):
            result += f"""
"node{node_index}" [
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


class DspNode(ABC):
    @abstractstaticmethod
    def outputs() -> list[str]:
        ...

    @abstractstaticmethod
    def inputs() -> list[str]:
        ...

    @abstractmethod
    def tick(self, dsp_graph: DspGraph):
        ...


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


if __name__ == "__main__":
    # s = SineOscilator()
    g = DspGraph()
    g.nodes.extend([SineOscilator(), ADSR(), SineOscilator(), Sum(), Output()])
    g.patch(0, "output", 1, "input")
    g.patch(1, "output", 2, "modulation")
    g.patch(0, "output", 3, "in_1")
    g.patch(2, "output", 3, "in_2")
    g.patch(3, "output", 4, "input")
    g.draw()
