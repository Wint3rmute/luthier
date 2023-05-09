import subprocess

import pytest


@pytest.fixture
def build_luthier() -> None:
    subprocess.check_output(["maturin", "develop"])
