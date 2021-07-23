import glob
import os
from ctypes import (
    cdll,
    Structure,
    c_int,
    c_long,
    POINTER,
    c_size_t,
    Union,
    c_void_p,
    c_char_p,
    CFUNCTYPE,
    Array,
    byref,
    pointer,
    create_string_buffer,
    c_ulong,
)

import pkg_resources

from .interface import ForfState, ForfProgram, Compiler
from .error import Error
from .func import ForfFunctionSet


class ForfEnv(Structure):
    pass


class ForfSlots(Structure):
    """
    struct slots {
      size_t  size;
      long   *slots;
    };
    """

    _fields_ = [("size", c_size_t), ("slots", POINTER(c_long))]

    @classmethod
    def new(cls, arr: Array) -> "ForfSlots":
        return ForfSlots(len(arr), arr)


class ForfEffect(Structure):
    """
    struct effect {
      long value;
      long slot;
    };
    """

    _fields_ = [("value", c_long), ("slot", c_long)]


class ForfEffects(Structure):
    """
    struct effects {
        size_t         size;
        struct effect *effects;
    };
    """

    _fields_ = [("size", c_size_t), ("slots", POINTER(ForfEffect))]

    @classmethod
    def new(cls, arr: Array) -> "ForfEffects":
        return ForfEffects(len(arr), arr)


class ForfFunc(Structure):
    """
    struct forf_func {
        struct slots   *input_slots;
        struct effects *effects;
        struct slots   *output_slots;
    };
    """

    _fields_ = [
        ("input_slots", POINTER(ForfSlots)),
        ("effects", POINTER(ForfEffects)),
        ("output_slots", POINTER(ForfSlots)),
    ]


ForfProc = CFUNCTYPE(None, POINTER(ForfEnv))

FORF_TYPE_FUNCTION = 2


class ForfValue(Structure):
    """
    struct forf_value {
      enum forf_value_type  type;
      union {
        forf_proc          *p;
        struct forf_func   *f;
        long                i;
      } v;
    };
    """

    class _U(Union):
        _fields_ = [("p", POINTER(ForfProc)), ("f", POINTER(ForfFunc)), ("i", c_long)]

    _fields_ = [("type", c_int), ("v", _U)]


class ForfStack(Structure):
    """
    struct forf_stack {
      size_t             size;
      size_t             top;
      struct forf_value *stack;
    };
    """

    _fields_ = [("size", c_size_t), ("top", c_size_t), ("stack", POINTER(ForfValue))]

    @classmethod
    def new(cls, arr):
        return cls(len(arr), 0, arr)


class ForfMemory(Structure):
    """
    struct forf_memory {
      size_t  size;
      long   *mem;
    };
    """

    _fields_ = [("size", c_size_t), ("mem", POINTER(c_long))]

    @classmethod
    def new(cls, arr):
        return cls(len(arr), arr)


class ForfLexicalEnv(Structure):
    """
    struct forf_lexical_env {
      char                  *name;
      enum forf_value_type  type;
      union {
          forf_proc          *p;
          struct forf_func   *f;
      } v;
    };
    """

    class _U(Union):
        _fields_ = [("p", POINTER(ForfProc)), ("f", POINTER(ForfFunc))]

    _fields_ = [("name", c_char_p), ("type", c_int), ("v", _U)]


"""
struct forf_env {
  enum forf_error_type     error;
  struct forf_lexical_env *lenv;
  struct forf_stack       *data;
  struct forf_stack       *command;
  struct forf_memory      *memory;
  struct forf_memory      *slots;
  long                    *rand_seed;
  void                    *udata;
};
"""
ForfEnv._fields_ = [
    ("error", c_int),
    ("lenv", POINTER(ForfLexicalEnv)),
    ("data", POINTER(ForfStack)),
    ("command", POINTER(ForfStack)),
    ("memory", POINTER(ForfMemory)),
    ("slots", POINTER(ForfMemory)),
    ("rand_seed", POINTER(c_ulong)),
    ("udata", c_void_p),
]


class CForfState(ForfState):
    def __init__(
        self,
        custom_function_set: ForfFunctionSet,
        rand_seed,
        command_stack_size=500,
        data_stack_size=200,
        memory_size=10,
    ):
        self._custom_function_set = custom_function_set
        self._cmdsvals = (ForfValue * command_stack_size)()
        self._datavals = (ForfValue * data_stack_size)()
        self._memvals = (c_long * memory_size)()
        self._slotvals = (c_long * custom_function_set.needed_slots)()

        self._cmd = ForfStack.new(self._cmdsvals)
        self._data = ForfStack.new(self._datavals)
        self._mem = ForfMemory.new(self._memvals)
        self._slots = ForfMemory.new(self._slotvals)

        self._rand_seed = c_ulong(rand_seed)

        self._env = ForfEnv(
            0,
            None,
            pointer(self._data),
            pointer(self._cmd),
            pointer(self._mem),
            pointer(self._slots),
            pointer(self._rand_seed),
            None,
        )

    def reset(self):
        # TODO: copy parsed code to cmd stack
        self._data.top = 0

    def get_mem(self):
        return self._memvals

    def get_error(self):
        return Error(self._env.error)

    def __getitem__(self, item: str) -> c_long:
        slot = self._custom_function_set.map_name_to_slot(item)
        return self._slotvals[slot]

    def __setitem__(self, key: str, value: c_long):
        slot = self._custom_function_set.map_name_to_slot(key)
        self._slotvals[slot] = value


class LibCForf:
    def __init__(self):
        print(pkg_resources.resource_listdir(__name__, ""))
        globs_to_try = [os.path.join(os.path.split(__file__)[0], "..", "forf.*.so")]
        for pattern in globs_to_try:
            res = glob.glob(pattern)
            if res:
                self._libcforf = cdll.LoadLibrary(res[0])
                break

    def run(self, code: bytes, lex_env: Array, state: CForfState):
        code_str = create_string_buffer(code)
        self._libcforf.forf_run(code_str, lex_env, byref(state._env))

    # def create_forf_state(self):
    #     state = CForfState()


class CForfInterpretable(ForfProgram):
    def __init__(self, libcforf: LibCForf, lex_env: Array, code: bytes):
        self._libcforf = libcforf
        self._lex_env = lex_env
        self._code = code

    def run(self, state: CForfState):
        self._libcforf.run(self._code, self._lex_env, state)


class CCompiler(Compiler):
    def __init__(
        self,
        custom_function_set: ForfFunctionSet = None,
        command_stack_size=500,
        data_stack_size=200,
        memory_size=10,
    ):
        super().__init__(
            custom_function_set, command_stack_size, data_stack_size, memory_size
        )
        self._libcforf = LibCForf()
        self._lex_env = self._create_lex_env(self._custom_function_set)

    def compile(self, code: str) -> ForfProgram:
        return CForfInterpretable(self._libcforf, self._lex_env, code.encode())

    def new_state(self, rand_seed: int) -> ForfState:
        # TODO: Finish me
        return CForfState(
            custom_function_set=self._custom_function_set, rand_seed=rand_seed
        )

    def _create_lex_env(self, custom_function_set: ForfFunctionSet):
        funcs = custom_function_set.funcs
        lex_env = (ForfLexicalEnv * (len(funcs) + 1))()

        # This is a list of structures that are passed by pointer to other structures.
        # The reason it exists to prevent Python GC from deleting the structures as they go out of scope.
        self._ref_holder = []

        for i, func in enumerate(funcs):
            in_arr = (c_long * len(func.input_slots))(*func.input_slots)
            in_slots = ForfSlots.new(in_arr)

            effects_arr = (ForfEffect * len(func.effects))()
            for j, effect in enumerate(func.effects):
                effects_arr[j].value = effect.value
                effects_arr[j].slot = effect.slot
            effects = ForfEffects.new(effects_arr)

            out_arr = (c_long * len(func.output_slots))(*func.output_slots)
            out_slots = ForfSlots.new(in_arr)
            self._ref_holder.extend(
                (in_arr, in_slots, effects_arr, effects, out_arr, out_slots)
            )
            f = ForfFunc(pointer(in_slots), pointer(effects), pointer(out_slots))
            self._ref_holder.append(f)

            func_name = create_string_buffer(func.token.encode())
            self._ref_holder.append(func_name)
            lex_env[i].name = func_name
            lex_env[i].type = FORF_TYPE_FUNCTION
            lex_env[i].v.f = pointer(f)
        lex_env[-1].name = None
        lex_env[-1].type = 0
        lex_env[-1].v.p = None
        return lex_env


def main():
    num_func_slots = 20
    c_size = 500
    d_size = 200
    mem_size = 10

    cmdsvals = (ForfValue * c_size)()
    datavals = (ForfValue * d_size)()
    memvals = (c_long * mem_size)()
    slotvals = (c_long * num_func_slots)()

    cmd = ForfStack.new(cmdsvals)
    data = ForfStack.new(datavals)
    mem = ForfMemory.new(memvals)
    slots = ForfMemory.new(slotvals)

    rand_seed = c_ulong(112)

    env = ForfEnv(
        0,
        None,
        pointer(data),
        pointer(cmd),
        pointer(mem),
        pointer(slots),
        pointer(rand_seed),
        None,
    )

    lex_env = (ForfLexicalEnv * 1)()
    lex_env[0].name = None
    lex_env[0].type = 0
    lex_env[0].v.p = None

    code = create_string_buffer(b"1 2 + 0 mset")

    cforf = LibCForf()
    cforf._libcforf.forf_run(code, lex_env, byref(env))
    print(env.error)
    print(list(memvals))


if __name__ == "__main__":
    main()
