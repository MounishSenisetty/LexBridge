import pytest

from lexbridge.resources import load_resources


@pytest.fixture(scope="session")
def res():
    """Builds the full fixture-based index once per test session (in a temp dir)."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        yield load_resources(build=True, out_dir=Path(d), use_network=False)
