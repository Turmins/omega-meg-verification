import pytest

from omega_dl.paths import inside_extension


def test_write_guard_rejects_external_target():
    with pytest.raises(ValueError):
        inside_extension(inside_extension.__globals__["extension_root"]().parent / "outside.txt")
