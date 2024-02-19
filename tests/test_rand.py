from ctypes import c_ulong
import pytest

from forf.rand import rand


@pytest.mark.parametrize("seed,n,chain", [
    (123, 100, [91, 82, 64])
])
def test_rand(seed, n, chain):
    seed = c_ulong(seed)
    n = c_ulong(n)
    for x in chain:
        seed, y = rand(seed, n)
        assert x == y.value
