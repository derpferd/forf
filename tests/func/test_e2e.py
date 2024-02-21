from typing import Dict, TypedDict

import pytest

from forf.cforf import CCompiler
from forf.func import CustomForfFunction, ForfFunctionSet, FunctionInput, FunctionOutput, FunctionSideEffect
from forf.interface import Compiler
from forf.error import Error


class TestOutput(TypedDict):
    mem: Dict[int, int]
    slots: Dict[str, int]


class TestCase(TypedDict):
    prog: str
    function_set: ForfFunctionSet
    slots_in: Dict[str, int]
    output: TestOutput


tests: Dict[str, TestCase] = {
    "simple in": {
        "prog": """1 foo""",
        "function_set": ForfFunctionSet(
            functions=[
                CustomForfFunction(token="foo", inputs=[FunctionInput("foo_value")])
            ]
        ),
        "slots_in": {},
        "output": {"slots": {"foo_value": 1}},
    },
    "simple out": {
        "prog": """foo 23 * 0 mset""",
        "function_set": ForfFunctionSet(
            functions=[
                CustomForfFunction(token="foo", outputs=[FunctionOutput("foo_value")])
            ]
        ),
        "slots_in": {"foo_value": 1},
        "output": {"mem": {0: 23}},
    },
    "simple effect": {
        "prog": """foo""",
        "function_set": ForfFunctionSet(
            functions=[
                CustomForfFunction(token="foo", side_effects=[FunctionSideEffect(1, "foo_value")])
            ]
        ),
        "slots_in": {},
        "output": {"slots": {"foo_value": 1}},
    },
}

COMPILERS = {CCompiler}


@pytest.mark.parametrize(
    "test_case, compiler",
    [
        pytest.param(case, compiler(custom_function_set=case["function_set"]), id=f"{compiler.__name__}-{name}")
        for name, case in tests.items()
        for compiler in COMPILERS
    ],
)
def test_e2e_happy_path(test_case: TestCase, compiler: Compiler):
    prog = test_case["prog"]
    output = test_case["output"]

    exe = compiler.compile(prog)
    state = compiler.new_state(rand_seed=123)
    for k, v in test_case['slots_in'].items():
        state[k] = v
    exe.run(state)

    assert state.get_error() == Error.NONE

    if "mem" in output:
        for mem_index, mem_value in output["mem"].items():
            actual = state.get_mem()[mem_index]
            assert (
                actual == mem_value
            ), f"Memory @{mem_index} is {actual} expected {mem_value}"

    if "slots" in output:
        for k, v in output['slots'].items():
            assert state[k] == v
