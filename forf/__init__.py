from .error import Error
from .func import (
    ForfFunctionSet,
    CustomForfFunction,
    FunctionInput,
    FunctionSideEffect,
    FunctionOutput,
)
from .interface import Compiler

import importlib

llvmlite_spec = importlib.util.find_spec("llvmlite")
if llvmlite_spec is not None:
    from .pyforf import InterpretableCompiler, ExecutableCompiler

from .cforf import CCompiler
