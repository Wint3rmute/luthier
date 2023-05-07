import numpy
import numpy.typing

F64NDArray = numpy.typing.NDArray[numpy.float64]

class DspGraph:
    def play(self, num_samples: int) -> F64NDArray: ...
