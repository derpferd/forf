from typing import TypedDict, Dict

import pytest

from forf import (
    InterpretableCompiler,
    ForfState,
    ExecutableCompiler,
    Compiler,
    CCompiler,
    Error,
)


class TestOutput(TypedDict):
    mem: Dict[int, int]


class TestCase(TypedDict):
    prog: str
    output: TestOutput


compilers = pytest.mark.parametrize(
    "compiler",
    [
        pytest.param(compiler(), id=compiler.__name__)
        for compiler in {InterpretableCompiler, ExecutableCompiler, CCompiler}
    ],
)

tests: Dict[str, TestCase] = {
    "no op": {"prog": """""", "output": {"mem": {}}},
    "simple store": {"prog": """1 0 mset""", "output": {"mem": {0: 1}}},
    "store negative number": {"prog": """-1 0 mset""", "output": {"mem": {0: -1}}},
    "storing result of op": {
        "prog": """58 58 * 0 mset""",
        "output": {"mem": {0: 3364}},
    },
    "dup": {"prog": """58 dup * 0 mset""", "output": {"mem": {0: 3364}}},
    "ifelse true": {
        "prog": """5 8 < { 50 8 + } { 50 8 - } ifelse 0 mset""",
        "output": {"mem": {0: 58}},
    },
    "ifelse false": {
        "prog": """8 5 < { 50 8 + } { 50 8 - } ifelse 0 mset""",
        "output": {"mem": {0: 42}},
    },
    "ifelse one operand out true": {
        "prog": """50 5 8 < { 8 + } { 8 - } ifelse 0 mset""",
        "output": {"mem": {0: 58}},
    },
    "ifelse one operand out false": {
        "prog": """50 8 5 < { 8 + } { 8 - } ifelse 0 mset""",
        "output": {"mem": {0: 42}},
    },
    "ifelse both operand out true": {
        "prog": """50 8 5 8 < { + } { - } ifelse 0 mset""",
        "output": {"mem": {0: 58}},
    },
    "ifelse both operand out false": {
        "prog": """50 8 8 5 < { + } { - } ifelse 0 mset""",
        "output": {"mem": {0: 42}},
    },
    "simple get": {
        "prog": """1 0 mset 0 mget 10 + 0 mset""",
        "output": {"mem": {0: 11}},
    },
    "trailing op true": {
        "prog": """1 2 < { 3 4 } { 5 6 } ifelse + 0 mset""",
        "output": {"mem": {0: 7}},
    },
    "trailing op false": {
        "prog": """1 2 > { 3 4 } { 5 6 } ifelse + 0 mset""",
        "output": {"mem": {0: 11}},
    },
    "macro ops": {
        "prog": "1 2 dup dup pop * exch dup * + 88 88 * < 0 mset",
        "output": {"mem": {0: 1}},
    },
    "test +-*/": {
        "prog": "4 3 - 1 + 4 * 2 / 0 mset",
        "output": {"mem": {0: 4}},
    },
    "test %": {
        "prog": "5 2 % 0 mset",
        "output": {"mem": {0: 1}},
    },
    "test &": {
        "prog": "255 7 & 0 mset",
        "output": {"mem": {0: 7}},
    },
    "test |": {
        "prog": "255 7 | 0 mset",
        "output": {"mem": {0: 255}},
    },
    # Nested ifelse
    # Variable number of values on stack after ifelse
}


@pytest.mark.parametrize(
    "test_case, compiler",
    [
        pytest.param(case, compiler(), id=f"{compiler.__name__}-{name}")
        for name, case in tests.items()
        for compiler in {InterpretableCompiler, ExecutableCompiler, CCompiler}
    ],
)
def test_e2e_happy_path(test_case: TestCase, compiler: Compiler):
    prog = test_case["prog"]
    output = test_case["output"]

    exe = compiler.compile(prog)
    state = ForfState.new()
    exe.run(state)

    for mem_index, mem_value in output["mem"].items():
        actual = state.mem[mem_index]
        assert (
            actual == mem_value
        ), f"Memory @{mem_index} is {actual} expected {mem_value}"


@compilers
def test_divide_by_zero(compiler):
    exe = compiler.compile("0 0 /")
    state = ForfState.new()
    exe.run(state)
    assert state.error == Error.DIVIDE_BY_ZERO


@compilers
def test_int_divide(compiler):
    exe = compiler.compile("7 6 / 0 mset")
    state = ForfState.new()
    exe.run(state)
    assert state.mem[0] == 1


@compilers
def test_memory_under_bounds(compiler):
    exe = compiler.compile("10 -1 mset")
    state = ForfState.new()
    exe.run(state)


# @pytest.mark.parametrize("test_case", [pytest.param(case, id=name)
#     for name, case in tests.items()
# ])
# def test_e2e_executable(test_case: TestCase):
#     prog = test_case["prog"]
#     output = test_case["output"]
#
#     exe = ExecutableCompiler().compile(prog)
#     state = ForfState.new()
#     exe.run(state)
#
#     for mem_index, mem_value in output['mem'].items():
#         actual = state.mem[mem_index]
#         assert actual == mem_value, f"Memory @{mem_index} is {actual} expected {mem_value}"
