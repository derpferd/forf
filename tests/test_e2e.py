from typing import TypedDict, Dict

import pytest

from forf import (
    InterpretableCompiler,
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
    "test + overflow": {
        "prog": "9223372036854775807 1 + 0 mset",
        "output": {"mem": {0: -9223372036854775808}},
    },
    "test - underflow": {
        "prog": "-9223372036854775808 1 - 0 mset",
        "output": {"mem": {0: 9223372036854775807}},
    },
    "test * overflow": {
        "prog": "1152921504606846976 8 * 0 mset",
        "output": {"mem": {0: -9223372036854775808}},
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
    "test ^": {
        "prog": "2 10 ^ 0 mset",
        "output": {"mem": {0: 8}},
    },
    "test <<": {
        "prog": "23 1 << 0 mset",
        "output": {"mem": {0: 46}},
    },
    "test << off the edge": {
        "prog": "4611686018427387904 2 << 0 mset",
        "output": {"mem": {0: 0}},
    },
    "test >> off the edge": {
        "prog": "1352 5 >> 0 mset",
        "output": {"mem": {0: 42}},
    },
    "test >> neg": {
        "prog": "-105 1 >> 0 mset",
        "output": {"mem": {0: -53}},
    },
    "test >": {
        "prog": "-1 2 > 2 2 > 4 3 > 2 mset 1 mset 0 mset",
        "output": {"mem": {0: 0, 1: 0, 2: 1}},
    },
    "test >=": {
        "prog": "-1 2 >= 2 2 >= 4 3 >= 2 mset 1 mset 0 mset",
        "output": {"mem": {0: 0, 1: 1, 2: 1}},
    },
    "test <": {
        "prog": "-1 2 < 2 2 < 4 3 < 2 mset 1 mset 0 mset",
        "output": {"mem": {0: 1, 1: 0, 2: 0}},
    },
    "test <=": {
        "prog": "-1 2 <= 2 2 <= 4 3 <= 2 mset 1 mset 0 mset",
        "output": {"mem": {0: 1, 1: 1, 2: 0}},
    },
    "test =": {
        "prog": "-1 2 = 2 2 = 4 3 = 2 mset 1 mset 0 mset",
        "output": {"mem": {0: 0, 1: 1, 2: 0}},
    },
    "test <>": {
        "prog": "-1 2 <> 2 2 <> 4 3 <> 2 mset 1 mset 0 mset",
        "output": {"mem": {0: 1, 1: 0, 2: 1}},
    }
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
    state = compiler.new_state(rand_seed=123)
    exe.run(state)

    for mem_index, mem_value in output["mem"].items():
        actual = state.get_mem()[mem_index]
        assert (
            actual == mem_value
        ), f"Memory @{mem_index} is {actual} expected {mem_value}"


@compilers
def test_divide_by_zero(compiler):
    exe = compiler.compile("0 0 /")
    state = compiler.new_state(rand_seed=123)
    exe.run(state)
    assert state.get_error() == Error.DIVIDE_BY_ZERO


@compilers
def test_int_divide(compiler):
    exe = compiler.compile("7 6 / 0 mset")
    state = compiler.new_state(rand_seed=123)
    exe.run(state)
    assert state.get_mem()[0] == 1


@compilers
def test_memory_under_bounds(compiler):
    exe = compiler.compile("10 -1 mset")
    state = compiler.new_state(rand_seed=123)
    exe.run(state)
    assert state.get_error() == Error.OVERFLOW


@compilers
def test_rand(compiler):
    exe = compiler.compile("100 random 100 random 100 random 0 mset 1 mset 2 mset")
    state = compiler.new_state(rand_seed=123)
    exe.run(state)
    assert state.get_mem()[0] == 64
    assert state.get_mem()[1] == 82
    assert state.get_mem()[2] == 91


@compilers
def test_rand_single(compiler):
    exe = compiler.compile("100 random 0 mset")
    state = compiler.new_state(rand_seed=123)
    exe.run(state)
    assert state.get_mem()[0] == 91
