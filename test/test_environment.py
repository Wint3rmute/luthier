def test_import_libraries_with_problematic_installation_procedures() -> None:
    import frechet_audio_distance
    import torch

    # This makes ruff happy :)
    assert frechet_audio_distance is not None
    assert torch is not None
