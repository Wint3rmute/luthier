import os

import pytest


@pytest.fixture
def build_luthier() -> None:
    os.system("maturin develop")
