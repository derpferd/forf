import json
import uuid
from abc import ABC, abstractmethod
from ctypes import POINTER, CFUNCTYPE, c_long, c_ulong, pointer
from dataclasses import dataclass
from subprocess import check_output
from typing import List

from llvmlite import ir

from base import (
    ForfState,
    remove_comments,
    parse,
    tokenize,
    ForfProg,
    create_execution_engine,
    ForfRand,
    make_rand,
    compile_ir,
    Error,
)
from func import CustomForfFunction

long = ir.IntType(64)


class ForfProgram(ABC):
    @abstractmethod
    def run(self, state: ForfState) -> ForfState:
        ...


class ForfInterpretable(ForfProgram):
    def __init__(self, prog: ForfProg, mem_size: int):
        self._prog = prog
        self._mem_size = mem_size

    def run(self, state: ForfState):
        if len(state.mem) != self._mem_size:
            raise ValueError("State memory size doesn't match code")
        try:
            self._prog.exec(state)
        except ZeroDivisionError:
            state.error = Error.DIVIDE_BY_ZERO


class ForfExecutable(ForfProgram):
    def __init__(self, cfunc, mem_size: int):
        self._cfunc = cfunc
        self._mem_size = mem_size

    def run(self, state: ForfState):
        if len(state.mem) != self._mem_size:
            raise ValueError("State memory size doesn't match code")
        res = self._cfunc(pointer(state.mem), pointer(state.rand_seed))
        state.error = Error(res)

        # assert res == 0


class CForf(ForfProgram):
    def __init__(self, code, mem_size: int):
        self._code = code
        self._mem_size = mem_size

    def run(self, state: ForfState):
        if len(state.mem) != self._mem_size:
            raise ValueError("State memory size doesn't match code")
        res_json = check_output(
            ["/home/derpferd/src/forf/cforf/cforfrunner", self._code]
            + [str(x) for x in state.mem]
        )
        res = json.loads(res_json)
        state.set_mem(res["mem"])
        state.error = Error(res["error_code"])


class Compiler(ABC):
    # def __init__(self, custom_functions: List[CustomForfFunction]):
    #     self._custom_functions = custom_functions

    @abstractmethod
    def compile(self, code: str, mem_size: int = 10) -> ForfProgram:
        ...


class CCompiler(ABC):
    def compile(self, code: str, mem_size: int = 10) -> ForfProgram:
        return CForf(code, mem_size)


class ExecutableCompiler(Compiler):
    def __init__(self):
        self._engine = create_execution_engine()

    def compile(self, code: str, mem_size: int = 10) -> ForfProgram:
        module = ir.Module(name=__file__)

        func_name = f"func_{uuid.uuid4().hex}"
        ast = parse(tokenize(remove_comments(code)))

        if ForfRand in ast:
            make_rand(module)

        long_array = ir.ArrayType(long, mem_size)
        fnty = ir.FunctionType(long, (long_array.as_pointer(), long.as_pointer()))
        func = ir.Function(module, fnty, name=func_name)

        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        ast.build(builder)
        compile_ir(self._engine, str(module))
        func_ptr = self._engine.get_function_address(func_name)
        CTypeArray = c_long * mem_size
        cfunc = CFUNCTYPE(c_long, POINTER(CTypeArray), POINTER(c_ulong))(func_ptr)
        return ForfExecutable(cfunc, mem_size=mem_size)


class InterpretableCompiler(Compiler):
    def compile(self, code: str, mem_size: int = 10) -> ForfProgram:
        prog = parse(tokenize(remove_comments(code)))
        return ForfInterpretable(prog=prog, mem_size=mem_size)
