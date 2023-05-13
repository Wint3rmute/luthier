from typing import Any


def test_graph_build(build_luthier: Any) -> None:
    from luthier import luthier

    luthier.DspGraph()


def test_graph_play(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    graph.play(100)


def test_graph_draw(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    sine = luthier.SineOscillator()
    graph.add_sine(sine)
    assert isinstance(graph.draw(), bytes)


def test_adsr(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    adsr = luthier.ADSR()
    adsr.input_attack = 1.0

    graph.add_adsr(adsr)
    graph.play(100)


def test_multiplier(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    multiplier = luthier.Multiplier()
    multiplier.input_scale = 1.0

    graph.add_multiplier(multiplier)
    graph.play(100)


def test_sum(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()

    sum = luthier.Sum()
    sum.input_in_1 = 1.0
    sum.input_in_2 = 2.0

    sum_id = graph.add_sum(sum)

    graph.patch(sum_id, "output_output", graph.speaker_node_id, "input_input")

    output = graph.play(10)
    assert output[5] == 3.0


def test_smoke_randomize_inputs(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    graph.randomize_inputs()


def test_harmonic_multiplier(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()

    multiplier = luthier.HarmonicMultiplier()
    multiplier.input_scale = 0.0
    multiplier.input_input = 1.0

    multiplier_id = graph.add_harmonic_multiplier(multiplier)
    graph.patch(multiplier_id, "output_output", graph.speaker_node_id, "input_input")

    output = graph.play(10)
    assert output[5] != 0.0


def test_num_inputs(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    assert graph.num_inputs() == 1
    assert len(graph.get_inputs()) == 1

    inputs = graph.get_inputs()
    inputs[0] = 0.15
    graph.set_inputs(inputs)

    inputs = graph.get_inputs()
    assert inputs[0] == 0.15


def test_low_pass(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    lowpass = luthier.LowPassFilter()
    graph.add_lowpass(lowpass)

    graph.play(100)


def test_square_osc(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    square = luthier.SquareOscillator()
    graph.add_square(square)

    graph.play(100)


def test_mverb(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    square = luthier.SquareOscillator()
    square_id = graph.add_square(square)

    reverb = luthier.Reverb()
    reverb_id = graph.add_reverb(reverb)

    graph.patch(reverb_id, "output_output", graph.speaker_node_id, "input_input")
    graph.patch(square_id, "output_output", reverb_id, "input_input")

    graph.play(100)
