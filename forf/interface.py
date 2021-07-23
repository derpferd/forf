from abc import abstractmethod, ABC

from .error import Error
from .func import ForfFunctionSet


class ForfState(ABC):
    """Base class for Forf State"""

    @abstractmethod
    def get_mem(self):
        ...

    @abstractmethod
    def get_error(self) -> Error:
        ...


class ForfProgram(ABC):
    @abstractmethod
    def run(self, state: ForfState) -> ForfState:
        ...


class Compiler(ABC):
    def __init__(
        self,
        custom_function_set: ForfFunctionSet = None,
        command_stack_size=500,
        data_stack_size=200,
        memory_size=10,
    ):
        if custom_function_set is None:
            custom_function_set = ForfFunctionSet([])
        self._custom_function_set = custom_function_set
        self._command_stack_size = command_stack_size
        self._data_stack_size = data_stack_size
        self._memory_size = memory_size

    @abstractmethod
    def compile(self, code: str) -> ForfProgram:
        ...

    @abstractmethod
    def new_state(self, rand_seed: int) -> ForfState:
        ...
