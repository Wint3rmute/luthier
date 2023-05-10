import subprocess

import pytest


@pytest.fixture(scope="session")
def build_luthier() -> None:
    subprocess.check_output(["maturin", "develop"])
