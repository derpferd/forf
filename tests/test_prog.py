import pytest

from forf.pyforf import *


@pytest.mark.parametrize(
    "prog,n,result",
    (
        (ForfProg.from_cmds(ForfValue(1)), 1, [ForfValue(1)]),
        (ForfProg.from_cmds(ForfValue(1), ForfValue(2)), 1, [ForfValue(2)]),
        (
            ForfProg.from_cmds(ForfValue(1), ForfValue(2)),
            2,
            [ForfValue(1), ForfValue(2)],
        ),
        (
            ForfProg.from_cmds(ForfValue(1), ForfMset("mset"), ForfValue(2)),
            2,
            [ForfValue(1), ForfValue(2)],
        ),
        (
            ForfProg.from_cmds(ForfValue(1), ForfMget("mget"), ForfValue(2)),
            2,
            [ForfVar(1), ForfValue(2)],
        ),
        (
            ForfProg.from_cmds(
                ForfValue(1), ForfBin("+", [ForfValue(2), ForfValue(3)]), ForfValue(4)
            ),
            2,
            [ForfBin("+", [ForfValue(2), ForfValue(3)]), ForfValue(4)],
        ),
        (
            ForfProg.from_cmds(
                ForfValue(1),
                ForfBin("+", [ForfMget("mget"), ForfValue(1)]),
                ForfValue(2),
            ),
            2,
            [ForfVar(1), ForfValue(2)],
        ),
    ),
)
def test_get_inputs(prog: ForfProg, n, result):
    assert prog.get_inputs(n) == result


@pytest.mark.parametrize(
    "prog,ns,results",
    (
        (
            ForfProg.from_cmds(ForfValue(1), ForfValue(2)),
            [1, 1],
            [[ForfValue(2)], [ForfValue(1)]],
        ),
        (
            ForfProg.from_cmds(
                ForfValue(1), ForfBin("+", [ForfValue(2), ForfValue(3)]), ForfValue(4)
            ),
            [2, 1],
            [
                [ForfBin("+", [ForfValue(2), ForfValue(3)]), ForfValue(4)],
                [ForfValue(1)],
            ],
        ),
    ),
)
def test_get_inputs_multiple_calls(prog: ForfProg, ns, results):
    assert [prog.get_inputs(n) for n in ns] == results


@pytest.mark.parametrize(
    "prog,ns",
    (
        (ForfProg.from_cmds(ForfValue(1), ForfValue(2)), [3]),
        (
            ForfProg.from_cmds(
                ForfValue(1), ForfBin("+", [ForfValue(2), ForfValue(3)]), ForfValue(4)
            ),
            [2, 2],
        ),
    ),
)
def test_get_inputs_pop_from_empty_list(prog: ForfProg, ns: List[int]):
    with pytest.raises(IndexError) as excinfo:
        for n in ns:
            prog.get_inputs(n)
    assert "pop from empty stack" in str(excinfo.value)
