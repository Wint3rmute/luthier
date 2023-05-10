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
