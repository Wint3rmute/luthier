from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from functools import cache, cached_property
from typing import Any, Callable, Optional

import audioflux as af
import librosa
import matplotlib.pyplot as plt
import numpy
import numpy.typing
from audioflux.display import fill_spec
from audioflux.type import SpectralFilterBankScaleType
from dtw import dtw
from IPython.display import Audio, display
from scipy.optimize import differential_evolution

from luthier import luthier

SAMPLE_RATE = 48000
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

    def get_input_by_index(self, input_index: int) -> float:
        return float(getattr(self.inputs, self.input_names()[input_index]))

    @abstractmethod
    def tick(self) -> None:
        ...


class Sample:
    def __init__(self, audio_buffer: AudioBuffer) -> None:
        self.buffer = audio_buffer
        # if sum(abs(audio_buffer)) == 0:
        #     self.buffer = audio_buffer
        # else:
        #     self.buffer = audio_buffer / numpy.linalg.norm(audio_buffer)
        #     self.buffer /= max(abs(self.buffer))
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
        ax.set_ylim(-1.1, 1.1)
        ax.plot(self.buffer[:num_samples])

    def show_player(self) -> None:
        """Display playble audio widget in Jupyter"""
        display(Audio(data=self.buffer, rate=SAMPLE_RATE))  # type: ignore

    def _fail_on_different_sample_lengths(self, other: "Sample") -> None:
        if len(self) != len(other):
            raise ValueError(
                f"Samples have different lengths, self: {len(self)}, other {len(other)}"
            )

    def mfcc_distance_with_dtw(self, other: "Sample", w: int = 1000) -> float:
        self._fail_on_different_sample_lengths(other)

        if w < 2:
            w = 2

        dist, cost, acc_cost, path = dtw(
            self.mfcc.T,
            other.mfcc.T,
            dist=lambda x, y: numpy.linalg.norm(x - y, ord=1),
            w=w,
        )
        return float(dist)

    def mfcc_distance_with_dtw_and_rms(self, other: "Sample") -> float:
        self._fail_on_different_sample_lengths(other)

        S, phase = librosa.magphase(librosa.stft(self.buffer))
        rms = librosa.feature.rms(S=S)

        mfcc_with_rms = numpy.concatenate((self.mfcc.T, rms.T), axis=1).T

        dist, cost, acc_cost, path = dtw(
            mfcc_with_rms.T,
            other.mfcc.T,
            dist=lambda x, y: x[-1] * numpy.linalg.norm(x[:-1] - y, ord=1),
        )
        return float(dist)

    def mfcc_distance(self, other: "Sample") -> float:
        self._fail_on_different_sample_lengths(other)
        # TODO: investigate whether this is definitely implemented right
        # as the optimisation doesnt work with this as target function
        return float(sum(abs(sum(abs(self.mfcc) - abs(other.mfcc)))))  # type: ignore

    def mfcc_distance_with_rms(self, other: "Sample") -> float:
        self._fail_on_different_sample_lengths(other)
        mfcc_distance_by_sample = abs(sum(abs(self.mfcc) - abs(other.mfcc)))
        S, phase = librosa.magphase(librosa.stft(self.buffer))
        rms = librosa.feature.rms(S=S)
        mfcc_distance_by_sample_by_rms = mfcc_distance_by_sample * rms
        sum_mfcc_distance = sum(sum(mfcc_distance_by_sample_by_rms))

        return float(sum_mfcc_distance)

    def spectrogram_distance(self, other: "Sample") -> float:
        self._fail_on_different_sample_lengths(other)
        _, self_spectro = self.spectrogram
        _, other_spectro = other.spectrogram

        return float(abs(sum(sum(abs(self_spectro) - abs(other_spectro)))))

    def spectrogram_distance_with_dtw(self, other: "Sample", w: int = 1000) -> float:
        self._fail_on_different_sample_lengths(other)

        if w < 2:
            w = 2

        _, self_spectro = self.spectrogram
        _, other_spectro = other.spectrogram

        dist, cost, acc_cost, path = dtw(
            self_spectro.T,
            other_spectro.T,
            dist=lambda x, y: numpy.linalg.norm(x - y, ord=1),
            w=w,
        )
        return float(dist)

    def plot_sound_overview(self, title: str = "Sound overview") -> None:
        fig, (ax, ax2, ax3) = plt.subplots(3)
        fig.suptitle(title)
        self.plot_waveform(ax, 1500)
        self.plot_spectrogram(ax2)
        self.plot_mfcc(ax3)
        self.show_player()
        plt.tight_layout()
        plt.show()


class DspGraphOptimizer:
    def __init__(
        self,
        graph_creation_function: Callable[[], luthier.DspGraph],
        difference_function: Callable[[Sample, Sample], float],
        target_audio: Sample,
        max_iterations: int = 100,
        population_size: int = 30,
        workers: int = -1,
    ) -> None:
        self.target_fun_values: list[float] = []
        self.best_parameters: list[numpy.typing.NDArray[numpy.float64]] = []
        self.best_sounds: list[Sample] = []

        self.workers = workers
        self.graph_creation_function = graph_creation_function
        self.difference_function = difference_function
        self.target_audio = target_audio
        self.max_iterations = max_iterations
        self.population_size = population_size

    def _callback(
        self, x: numpy.typing.NDArray[numpy.float64], convergence: Any
    ) -> None:
        target_fun = self._target_function(x)
        sample = self._make_sound(x)

        self.best_sounds.append(sample)
        self.target_fun_values.append(target_fun)

        # clear_output()
        # show_comparison(x)
        # plt.plot(self.target_fun_values)
        # plt.scatter(self.max_iterations, 0)
        # plt.title(f"Training progress (problem size {len(x)})")
        # plt.xlabel("Iteration number")
        # plt.ylabel("MFCC distance")
        # plt.show()
        self.target_fun_values.append(target_fun)
        self.best_parameters.append(x)

    def _make_sound(self, inputs: numpy.typing.NDArray[numpy.float64]) -> Sample:
        graph = self.graph_creation_function()
        graph.set_inputs(inputs)

        return Sample(graph.play(len(self.target_audio)))

    def _target_function(self, inputs: numpy.typing.NDArray[numpy.float64]) -> float:
        generated_audio = self._make_sound(inputs)
        try:
            dist = self.difference_function(self.target_audio, generated_audio)
        except Exception as e:
            raise ValueError(f"Exception for params {list(inputs)}") from e
        return float(dist)

    def optimize(self) -> None:
        num_inputs = self.graph_creation_function().num_inputs()

        differential_evolution(
            self._target_function,
            [(-1.0, 1.0) for i in range(num_inputs)],
            workers=self.workers,
            maxiter=self.max_iterations,
            popsize=self.population_size,
            polish=False,
            disp=False,
            callback=self._callback,
        )


if __name__ == "__main__":

    def graph_create_fun() -> luthier.DspGraph:
        return luthier.DspGraph()

    def mfcc_dtw_distance(target: Sample, other: Sample) -> float:
        return target.mfcc_distance_with_dtw(other)

    o = DspGraphOptimizer(
        graph_create_fun,
        mfcc_dtw_distance,
        Sample(numpy.zeros(1000)),
    )
    o.optimize()
