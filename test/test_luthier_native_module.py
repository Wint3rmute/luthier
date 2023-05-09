from typing import Any


def test_graph_build(build_luthier: Any) -> None:
    from luthier import luthier
    luthier.DspGraph()


def test_graph_play(build_luthier: Any) -> None:
    from luthier import luthier

    graph = luthier.DspGraph()
    graph.play(100)
