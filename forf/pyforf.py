import copy
import re
import abc
import uuid
from enum import Enum, auto
from random import randint

import llvmlite.binding as llvm
from llvmlite import ir
from dataclasses import dataclass
from ctypes import CFUNCTYPE, c_long, POINTER, pointer, c_ulong, Array
from typing import List, Dict, Optional, Union, Tuple

from llvmlite.ir import Instruction

from .error import Error
from .func import ForfFunctionSet
from .interface import ForfState, ForfProgram, Compiler

TAB = "  "

long = ir.IntType(64)


@dataclass
class PyForfState(ForfState):
    cmd: Array
    data: Array
    mem: Array
    slots: Array
    rand_seed: c_ulong
    error: Error = Error.NONE

    @classmethod
    def new(
        cls, func_slots, mem_size=10, rand_seed: Optional[Union[c_ulong, int]] = None
    ):
        cmd_size = 500
        data_size = 200
        _cmd = (c_long * cmd_size)(*[0 for _ in range(cmd_size)])
        _data = (c_long * data_size)(*[0 for _ in range(data_size)])
        _mem = (c_long * mem_size)(*[0 for _ in range(mem_size)])
        _slots = (c_long * mem_size)(*[0 for _ in range(func_slots)])
        if rand_seed is None:
            rand_seed = c_ulong(randint(0, 2 ** 40))
        if isinstance(rand_seed, int):
            rand_seed = c_ulong(rand_seed)
        return cls(cmd=_cmd, data=_data, mem=_mem, slots=_slots, rand_seed=rand_seed)

    def get_mem(self):
        return self.mem

    def get_error(self):
        return self.error

    def set_mem(self, mem: List[int]):
        assert len(mem) == len(self.mem)
        for i, x in enumerate(mem):
            self.mem[i] = x

    # data: List[int]
    # mem: List[int]
    # rand_seed: int
    # state: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    #
    # data_index: int = 0
    # code_index: int = 0
    #
    # @classmethod
    # def new(cls, data_size=200, mem_size=10):
    #     _data = [0 for _ in range(data_size)]
    #     _mem = [0 for _ in range(mem_size)]
    #     seed = random.randint(0, 2**40)
    #     return cls(data=_data, mem=_mem, rand_seed=seed)


class ForfCmd(abc.ABC):
    OPS = {}
    INPUTS = 0
    OUTPUTS = 0
    ORDER_SENSITIVE = False

    def __init__(
        self, token: Union[str, int], children: Optional[List["ForfCmd"]] = None
    ):
        self._token = token
        self._children = children

    @abc.abstractmethod
    def exec(self, state: PyForfState):
        ...

    @abc.abstractmethod
    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        ...

    @property
    def will_return_value(self) -> bool:
        return True

    @property
    def is_order_sensitive(self) -> bool:
        if self.ORDER_SENSITIVE:
            return True
        if self._children and any(c.is_order_sensitive for c in self._children):
            return True
        return False

    def replace(self, old, new) -> "ForfCmd":
        if old == self:
            return new
        if self._children:
            self._children = list(map(lambda x: x.replace(old, new), self._children))
        return self

    def output(self) -> List["ForfCmd"]:
        return [self]

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and isinstance(other, ForfCmd)
            and self._token == other._token
            and self._children == other._children
        )

    def __str__(self):
        children_str = ""
        if self._children:
            children_str = list(str(c) for c in self._children)
        return f"{self.__class__.__name__}({children_str})"

    def __repr__(self):
        return str(self)

    def __contains__(self, item):
        if isinstance(item, type):
            if isinstance(self, item):
                return True
            if self._children:
                return any(item in child for child in self._children)
            return False
        raise NotImplementedError()


class ForfVar(ForfCmd):
    INPUTS = 0
    OUTPUTS = 1

    def exec(self, state: PyForfState):
        return state.data[self._token]
        # raise NotImplementedError()
        # return self._token

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        return variables[self._token]
        # raise NotImplementedError()
        # return long(self._token)

    def __str__(self):
        return f"Var({self._token})"


class ForfValue(ForfCmd):
    INPUTS = 0
    OUTPUTS = 1

    def exec(self, state: PyForfState):
        return self._token

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        return long(self._token)

    def __str__(self):
        return f"Value({self._token})"


class ForfUni(ForfCmd):
    """1 input => 1 output"""

    OPS = {
        "~": "~",
        "!": lambda x: not x,
        "abs": lambda x: abs(x),
    }
    INPUTS = 1
    OUTPUTS = 1

    def exec(self, state: PyForfState):
        return self.OPS[self._token](self._child.exec())

    @property
    def _child(self):
        (child,) = self._children
        return child

    def __str__(self):
        return f"Uni({self._token}, {self._child})"


class ForfBin(ForfCmd):
    """2 inputs => 1 output"""

    OPS = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x // y,
        "%": lambda x, y: x % y,
        "&": lambda x, y: x & y,
        "|": lambda x, y: x | y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
        ">": lambda x, y: int(x > y),
        ">=": lambda x, y: int(x >= y),
        "<": lambda x, y: int(x < y),
        "<=": lambda x, y: int(x <= y),
        "=": lambda x, y: int(x == y),
        "<>": lambda x, y: int(x != y),
    }
    INPUTS = 2
    OUTPUTS = 1

    def exec(self, state: PyForfState):
        b1, b2 = self._children
        return self.OPS[self._token](b1.exec(state), b2.exec(state))

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        b1, b2 = self._children
        ir_1 = b1.build(builder, variables)
        ir_2 = b2.build(builder, variables)

        simple_ops_to_builder = {
            "+": lambda: builder.add(ir_1, ir_2),
            "-": lambda: builder.sub(ir_1, ir_2),
            "*": lambda: builder.mul(ir_1, ir_2),
            "%": lambda: builder.srem(ir_1, ir_2),
            "&": lambda: builder.and_(ir_1, ir_2),
            "|": lambda: builder.or_(ir_1, ir_2),
            "^": lambda: builder.xor(ir_1, ir_2),
            "<<": lambda: builder.shl(ir_1, ir_2),
            ">>": lambda: builder.ashr(ir_1, ir_2),
        }

        if self._token in simple_ops_to_builder:
            return simple_ops_to_builder[self._token]()
        if self._token == "/":
            pred = builder.icmp_signed("==", ir_2, long(0))
            with builder.if_then(pred):
                builder.ret(long(Error.DIVIDE_BY_ZERO.value))
            return builder.sdiv(ir_1, ir_2)
        if self._token in {"<", ">", ">=", "<=", "=", "<>"}:
            mapped_token = {"=": "==", "<>": "!="}.get(self._token, self._token)
            return builder.select(
                builder.icmp_signed(mapped_token, ir_1, ir_2), long(1), long(0)
            )
        raise NotImplementedError

    def __str__(self):
        b1, b2 = self._children
        return f"Bin({b1}, {self._token}, {b2})"


class ForfPop(ForfCmd):
    """1 input => 0 output"""

    OPS = {"pop"}
    INPUTS = 1
    OUTPUTS = 0

    def exec(self, state: PyForfState):
        raise ValueError("This value is evaluated at compile time!")

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        raise ValueError("This value is evaluated at compile time!")

    def __str__(self):
        (b1,) = self._children
        return f"Pop({b1})"

    def output(self) -> List["ForfCmd"]:
        return []


class ForfDup(ForfCmd):
    """1 input => 2 output"""

    OPS = {"dup"}
    INPUTS = 1
    OUTPUTS = 2

    def exec(self, state: PyForfState):
        raise ValueError("This value is evaluated at compile time!")

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        raise ValueError("This value is evaluated at compile time!")

    def __str__(self):
        (b1,) = self._children
        return f"Dup({b1})"

    def output(self) -> List["ForfCmd"]:
        (b1,) = self._children
        return [b1, copy.deepcopy(b1)]


class ForfExch(ForfCmd):
    """2 input => 2 output"""

    OPS = {"exch"}
    INPUTS = 2
    OUTPUTS = 2

    def exec(self, state: PyForfState):
        raise ValueError("This value is evaluated at compile time!")

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        raise ValueError("This value is evaluated at compile time!")

    def output(self) -> List["ForfCmd"]:
        b1, b2 = self._children
        return [b2, b1]


class ForfMset(ForfCmd):
    """2 input => 0 output"""

    OPS = {"mset"}
    INPUTS = 2
    OUTPUTS = 0
    ORDER_SENSITIVE = True

    @property
    def will_return_value(self) -> bool:
        return False

    def exec(self, state: PyForfState):
        slot = self._slot.exec(state)
        if slot < 0:
            raise IndexError("Index out of range")
        state.mem[slot] = self._value.exec(state)

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        ir_slot = self._slot.build(builder, variables)
        pred = builder.icmp_signed("<", ir_slot, long(0))
        with builder.if_then(pred):
            builder.ret(long(Error.OVERFLOW.value))

        mem, _ = builder.function.args
        x = builder.gep(mem, (long(0), ir_slot))
        builder.store(self._value.build(builder, variables), x)

    @property
    def _slot(self):
        _, slot = self._children
        return slot

    @property
    def _value(self):
        value, _ = self._children
        return value

        # raise NotImplementedError

    # def output(self) -> List["ForfCmd"]:
    #     raise [self]


class ForfMget(ForfCmd):
    """1 input => 1 output"""

    OPS = {"mget"}
    INPUTS = 1
    OUTPUTS = 1
    ORDER_SENSITIVE = True

    def exec(self, state: PyForfState):
        (b1,) = self._children
        return state.mem[b1.exec(state)]

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        (b1,) = self._children
        mem, _ = builder.function.args
        x = builder.gep(mem, (long(0), b1.build(builder, variables)))
        return builder.load(x)


# class ForfIf(ForfCmd):
#     """2 input => 1 output"""
#
#     OPS = {"if"}
#     INPUTS = 2
#     OUTPUTS = 1
#
#     def exec(self, state: ForfProgState):
#         ...
#
#     def __str__(self):
#         block = "\n".join(f"{TAB}{s}" for s in str(self._children[0]).split("\n"))
#         return f"If({self._children[1]}) {{\n{block}\n}}"


class ForfIfElse(ForfCmd):
    """? input => ? output"""

    OPS = {"ifelse"}
    INPUTS = float("inf")
    OUTPUTS = float("inf")

    def __init__(self, token: str, cond: ForfCmd, if_blocks, else_blocks):
        super().__init__(token)
        self._condition = cond
        self._if_blocks = if_blocks
        self._else_blocks = else_blocks

    def exec(self, state: PyForfState):
        blocks = self._else_blocks
        if self._condition.exec(state):
            blocks = self._if_blocks

        for block in blocks:
            block.cmd.exec(state)

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        pred = builder.icmp_signed(
            "==", self._condition.build(builder, variables), long(0)
        )

        with builder.if_else(pred) as (then, otherwise):
            with then:
                for block in self._else_blocks:
                    block.cmd.build(builder, variables)
            with otherwise:
                for block in self._if_blocks:
                    block.cmd.build(builder, variables)
        # return builder.select(
        #     pred,
        #     self._if_blocks[0].build(builder),
        #     self._else_blocks[0].build(builder)
        # )

    @property
    def _children(self):
        return (
            [self._condition]
            + [b.cmd for b in self._if_blocks]
            + [b.cmd for b in self._else_blocks]
        )

    @_children.setter
    def _children(self, v):
        ...

    # @property
    # def _condition(self):
    #     return self._children[0]
    #
    # @property
    # def _if_block(self):
    #     return self._children[1]
    #
    # @property
    # def _else_block(self):
    #     return self._children[2]

    def __str__(self):
        statement = self._condition
        if_block = "\n".join(f"{TAB}{s}" for s in str(self._if_blocks).split("\n"))
        else_block = "\n".join(f"{TAB}{s}" for s in str(self._else_blocks).split("\n"))
        return f"If({statement}) {{\n{if_block}\n}}\nElse\n{{\n{else_block}\n}}"


class ForfFuncType(Enum):
    GETTER = auto()
    SETTER = auto()


class ForfFunc(ForfCmd):
    ORDER_SENSITIVE = True


"""
func_data

Input           : input -> func_data
Side effects    : const -> func_data
Output          : func_data -> stack  (only allow one output)

"""


class ForfRand(ForfCmd):
    OPS = {"random"}
    INPUTS = 1
    OUTPUTS = 1
    ORDER_SENSITIVE = True

    def exec(self, state: PyForfState):
        raise NotImplementedError

    def build(self, builder: ir.IRBuilder, variables: Dict[int, Instruction]):
        _, seed = builder.function.args
        b1 = self._children[0]
        ir_1 = b1.build(builder, variables)
        rand = builder.module.get_global("rand")
        return builder.call(rand, (seed, ir_1))


# class ForfFunc(ForfCmd):
#     token: str = "fire-ready?"
#     type: ForfFuncType = ForfFuncType.GETTER
#     def __init__(self):
#         ...


class ForfProg:
    @dataclass
    class Block:
        cmd: ForfCmd
        assigned_var: Optional[int] = None
        used: bool = False

    def __init__(self, blocks: List["ForfProg.Block"] = None, num_vars: int = 0):
        self._blocks = []
        if blocks:
            self._blocks = copy.deepcopy(blocks)
        self._last_i = num_vars
        self._frozen = False
        self._no_pop_before_i = -1

    def __str__(self):
        return f"ForfProg(last_i: {self._last_i} Blocks: {self._blocks})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self._blocks == other._blocks
            and self._last_i == other._last_i
        )

    def __contains__(self, item):
        if isinstance(item, type):
            if isinstance(self, item):
                return True
            if self._blocks:
                return any(item in block.cmd for block in self._blocks)
            return False
        raise NotImplementedError()

    def exec(self, state: PyForfState):
        if not self._frozen:
            raise ValueError("Prog must be frozen before execution.")
        for block in self._blocks:
            val = block.cmd.exec(state)
            if block.assigned_var is not None:
                # TODO: bounds checking
                state.data[block.assigned_var] = val

    def add_block(self, cmd: ForfCmd):
        if self._frozen:
            raise ValueError("This Prog is frozen so can't be modified")
        block = ForfProg.Block(cmd=cmd, assigned_var=self._last_i)
        self._last_i += 1
        self._blocks.append(block)

    def set_stable(self):
        self._no_pop_before_i = self._last_i

    # def merge(self, other: 'ForfProg'):
    #     new = ForfProg()
    #     new._frozen = True
    #
    #     raise NotImplementedError
    @staticmethod
    def merge_progs_if_else(
        token: str, cond_cmd: ForfCmd, if_prog: "ForfProg", else_prog: "ForfProg"
    ) -> "ForfProg":
        new_prog = ForfProg()
        # TODO: Optimize (reduce duplicate code and variables)
        if isinstance(cond_cmd, ForfVar):
            raise NotImplementedError
        else:
            new_prog.add_block(
                ForfIfElse(
                    token=token,
                    cond=cond_cmd,
                    if_blocks=if_prog._blocks,
                    else_blocks=else_prog._blocks,
                )
            )
            new_prog._last_i = max(if_prog._last_i, else_prog._last_i)
        return new_prog
        # new_prog.
        # cond_cmd

    def compress(self):
        new = ForfProg()
        new._frozen = True
        var_index_map_old_to_new = {}
        for block in self._blocks:
            if block.used:
                var_index_map_old_to_new[block.assigned_var] = len(
                    var_index_map_old_to_new
                )
                new._blocks.append(
                    ForfProg.Block(
                        cmd=block.cmd,
                        assigned_var=var_index_map_old_to_new[block.assigned_var],
                        used=True,
                    )
                )
            else:
                new._blocks.append(ForfProg.Block(cmd=block.cmd))

        def replace_var_indexes(block: ForfProg.Block):
            for old_i, new_i in var_index_map_old_to_new.items():
                block.cmd = block.cmd.replace(ForfVar(old_i), ForfVar(new_i))
            return block

        new._blocks = list(map(lambda b: replace_var_indexes(b), new._blocks))
        new._last_i = len(var_index_map_old_to_new)
        return new
        # if all(block.used==False for block in self._blocks):
        #     new = ForfProg()
        #     new._frozen = True
        # reduce the number of variables used
        # raise NotImplementedError

    def get_inputs(self, num: int) -> List[ForfCmd]:
        if self._frozen:
            raise ValueError("This Prog is frozen so can't be modified")
        inputs = []
        i = len(self._blocks)
        while len(inputs) < num and i > 0:
            i -= 1
            cur = self._blocks[i]
            if cur.used:
                continue
            assert not isinstance(cur.cmd, ForfIfElse)
            if cur.cmd.OUTPUTS == 1:
                if cur.cmd.is_order_sensitive:
                    inputs.append(ForfVar(cur.assigned_var))
                    cur.used = True
                else:
                    inputs.append(cur.cmd)
                    self._blocks.pop(i)
            else:
                assert cur.cmd.OUTPUTS == 0
        if len(inputs) != num:
            raise IndexError("pop from empty stack")
        return inputs[::-1]

    @classmethod
    def from_cmds(cls, *cmds):
        prog = ForfProg()
        for cmd in cmds:
            prog.add_block(cmd)
        return prog

    def build(self, builder: ir.IRBuilder):
        variables = {}
        for block in self._blocks:
            res = block.cmd.build(builder, variables)
            if block.assigned_var is not None:
                variables[block.assigned_var] = res
        builder.ret(long(0))

    # def __init__(self, cmd_stack: List[ForfCmd]):
    #     ...

    # @classmethod
    # def from_string(self, code: str):
    #     ...


def remove_comments(code) -> str:
    return re.sub(r"\(.*?\)", "", code, flags=re.DOTALL)


def tokenize(code) -> List[str]:
    return list(re.split(r"\s+", code))


TOKEN_TO_CLASS = dict()
for c in ForfCmd.__subclasses__():
    for op in c.OPS:
        TOKEN_TO_CLASS[op] = c


def _parse_block(
    start: int, tokens: List[str], list_of_cmd: List[ForfCmd]
) -> Tuple[int, List[ForfCmd]]:
    i = start
    while tokens[i] != "}":
        i, list_of_cmd = _parse_token(i, tokens, list_of_cmd)
    return i + 1, list_of_cmd


def _parse_if(
    start: int, tokens: List[str], orig_of_cmd: List[ForfCmd]
) -> Tuple[int, ForfIfElse]:
    cond = orig_of_cmd.pop()
    block_1_list = copy.deepcopy(orig_of_cmd)
    block_2_list = copy.deepcopy(orig_of_cmd)
    i = start
    i, block_1_list = _parse_block(i, tokens, block_1_list)
    if tokens[i] == "{":
        i += 1
        i, block_2_list = _parse_block(i, tokens, block_2_list)
    token = tokens[i]
    i += 1
    stop_index = i
    while stop_index < len(tokens) and tokens[stop_index] != "}":
        stop_index += 1
    block1_i = i
    while block1_i < stop_index:
        block1_i, block_1_list = _parse_token(block1_i, tokens, block_1_list)
    block2_i = i
    while block2_i < stop_index:
        block2_i, block_2_list = _parse_token(block2_i, tokens, block_2_list)

    return stop_index, ForfIfElse(token, [cond, block_1_list, block_2_list])
    # list_of_cmd = []
    # i = start
    # while tokens[i] != '}':
    #     i, cmds = _parse_token(i, tokens)
    #     list_of_cmd.extend(cmds)
    #     i += 1
    # return i + 1, list_of_cmd


def _parse_block_prog(
    start: int, tokens: List[str], prog: ForfProg
) -> Tuple[int, ForfProg]:
    i = start
    while tokens[i] != "}":
        i, prog = _parse_token_prog(i, tokens, prog)
    return i + 1, prog


def _parse_if_prog(
    start: int, tokens: List[str], prog: ForfProg
) -> Tuple[int, ForfProg]:
    cond = prog.get_inputs(1)[0]
    block_1_prog = copy.deepcopy(prog)
    block_2_prog = copy.deepcopy(prog)
    i = start
    i, block_1_prog = _parse_block_prog(i, tokens, block_1_prog)
    if tokens[i] == "{":
        i += 1
        i, block_2_prog = _parse_block_prog(i, tokens, block_2_prog)
    token = tokens[i]
    i += 1
    stop_index = i
    while stop_index < len(tokens) and tokens[stop_index] != "}":
        stop_index += 1
    block1_i = i
    while block1_i < stop_index:
        block1_i, block_1_prog = _parse_token_prog(block1_i, tokens, block_1_prog)
    block2_i = i
    while block2_i < stop_index:
        block2_i, block_2_prog = _parse_token_prog(block2_i, tokens, block_2_prog)

    # ForfIfElse(token, [cond, block_1_list, block_2_list])
    new_prog = ForfProg.merge_progs_if_else(
        token=token, cond_cmd=cond, if_prog=block_1_prog, else_prog=block_2_prog
    )
    return stop_index, new_prog


# def _get_inputs_from_list(num: int, list_of_cmd) -> List[ForfCmd]:
#     inputs = []
#     i = len(list_of_cmd)
#     while len(inputs) < num and i > 0:
#         i -= 1
#         cur = list_of_cmd[i]
#         assert not isinstance(cur, ForfIfElse)
#         if cur.OUTPUTS == 1:
#             inputs.append(cur)
#             list_of_cmd.pop(i)
#         else:
#             assert cur.OUTPUTS == 0
#     if len(inputs) != num:
#         raise IndexError('pop from empty stack')
#     return inputs[::-1]


# def _parse_token(start, tokens, list_of_cmd) -> Tuple[int, List[ForfCmd]]:
#     token = tokens[start]
#     i = start + 1
#
#     if re.fullmatch(r"-?0\d+", token):
#         list_of_cmd.append(ForfValue(int(token, 8)))
#     elif re.fullmatch(r"-?\d+", token):
#         list_of_cmd.append(ForfValue(int(token)))
#     elif re.fullmatch(r"-?0x\d+", token):
#         list_of_cmd.append(ForfValue(int(token, 16)))
#     elif token in TOKEN_TO_CLASS:
#         cls = TOKEN_TO_CLASS[token]
#         # children = [list_of_cmd.pop() for _ in range(cls.INPUTS)][::-1]
#         children = _get_inputs_from_list(cls.INPUTS, list_of_cmd)
#         list_of_cmd.extend(cls(token, children).output())
#     elif token == "{":
#         i, cmd = _parse_if(i, tokens, list_of_cmd)
#         list_of_cmd = [cmd]
#
#     return i, list_of_cmd


def _parse_token_prog(start, tokens, prog: ForfProg) -> Tuple[int, ForfProg]:
    token = tokens[start]
    i = start + 1

    if re.fullmatch(r"-?0\d+", token):
        prog.add_block(ForfValue(int(token, 8)))
    elif re.fullmatch(r"-?\d+", token):
        prog.add_block(ForfValue(int(token)))
    elif re.fullmatch(r"-?0x\d+", token):
        prog.add_block(ForfValue(int(token, 16)))
    elif token in TOKEN_TO_CLASS:
        cls = TOKEN_TO_CLASS[token]
        # children = [list_of_cmd.pop() for _ in range(cls.INPUTS)][::-1]
        children = prog.get_inputs(cls.INPUTS)
        for cmd in cls(token, children).output():
            prog.add_block(cmd)
    elif token == "{":
        i, prog = _parse_if_prog(i, tokens, prog)
        # list_of_cmd = [cmd]

    return i, prog


def parse(tokens: List[str]):
    i = 0
    prog = ForfProg()
    # list_of_cmd = []
    while i < len(tokens):
        i, prog = _parse_token_prog(i, tokens, prog)
        # list_of_cmd.extend(cmds)
    return prog.compress()
    # i = 0
    # while i < len(tokens):
    #     token = tokens[i]
    #     i += 1
    #     if re.fullmatch(r'-?0\d+', token):
    #         list_of_cmd.append(ForfValue(int(token, 8)))
    #     elif re.fullmatch(r'-?\d+', token):
    #         list_of_cmd.append(ForfValue(int(token)))
    #     elif re.fullmatch(r'-?0x\d+', token):
    #         list_of_cmd.append(ForfValue(int(token, 16)))
    #     elif token in token_to_class:
    #         cls = token_to_class[token]
    #         children = [list_of_cmd.pop() for _ in range(cls.INPUTS)][::-1]
    #         list_of_cmd.extend(cls(token, children).output())
    #     elif token == '{':
    #         i, block = _parse_block(i, tokens)
    #         list_of_cmd.extend(block)
    # assert len(list_of_cmd) == 1
    # return list_of_cmd[0]


def exec(code):
    state = PyForfState.new()
    res = []
    for cmd in parse(tokenize(remove_comments(code))):
        exec_res = cmd.exec(state)
        # if isinstance(exec_res, list):
        #     res.extend(exec_res)
        # else:
        #     res.append(exec_res)
    return state


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one


def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def comp(module, code):
    ast = parse(tokenize(remove_comments(code)))
    if any(ForfRand in b for b in ast):
        make_rand(module)

    long_array = ir.ArrayType(long, 10)
    fnty = ir.FunctionType(long, (long_array.as_pointer(), long.as_pointer()))
    func = ir.Function(module, fnty, name="fptest")

    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    # result = builder.add(long(1), long(2))
    result = ast[0].build(builder)
    # x = builder.alloca(c_ulong)
    # seed = c_ulong(123)
    # module.get_global()
    # n = 100
    # _, seed = builder.function.args
    # result = builder.call(rand, (seed, long(100)))
    # a, = func.args
    # a = builder.alloca(long)
    # builder.store(long(123), a)
    # builder.insert_value(a, long(123), 2)
    # x = builder.gep(a, (long(0), long(4)))
    # builder.store(long(123), x)
    # result = builder.load(x)
    if not result:
        result = long(0)
    builder.ret(result)
    return module


def compile_ir(engine, llvm_ir):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    # Create a LLVM module object from the IR
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    # return mod


def main():
    engine = create_execution_engine()
    # with open('../rand.ll') as fp:
    #     rand_mod_ir = fp.read()
    # rand_mod = compile_ir(engine, rand_mod_ir)
    module = ir.Module(name=__file__)

    # print(str(rand_ir))
    # compile_ir(engine, str(rand_ir))

    # module_ir = comp(module, "2 mget 10 + 2 mset", rand=rand)
    module_ir = comp(module, "1 random 5 mset")
    print(str(module_ir))
    compile_ir(engine, str(module_ir))

    # rand_func_ptr = engine.get_function_address("rand")
    # cfunc = CFUNCTYPE(c_long, POINTER(c_ulong), c_long)(rand_func_ptr)
    # seed = c_ulong(123)
    # n = 100
    # res = cfunc(pointer(seed), n)
    # print(f'rand() = {res}')
    # print(f'seed = {seed}')
    # print()

    func_ptr = engine.get_function_address("fptest")
    IntArray10 = c_long * 10
    cfunc = CFUNCTYPE(c_long, POINTER(IntArray10), POINTER(c_ulong))(func_ptr)
    arr = IntArray10(1, 2, 3, 4, 5, 6)
    seed = c_ulong(123)
    res = cfunc(pointer(arr), pointer(seed))
    print("fptest(...) =", res)
    print(list(arr))

    # rand_func_ptr = engine.get_function_address("rand")
    # cfunc = CFUNCTYPE(c_long, POINTER(c_ulong), c_long)(rand_func_ptr)
    # seed = c_ulong(123)
    # n = 100
    # res = cfunc(pointer(seed), n)
    # print(f'rand() = {res}')
    # print(f'seed = {seed}')
    # res = cfunc(pointer(seed), n)
    # print(f'rand() = {res}')
    # print(f'seed = {seed}')
    # print("hi")
    # print()
    # s = "58 58 *"
    # # s = '5 8 < { 50 8 + } { 50 8 - } ifelse 5 ! 1 2 + dup 3 4 + pop'
    # # s = '4 dup * dup 100 < { pop 0 } if'
    # code = s
    #
    # print(f"=== Before Output ===")
    # for b in parse(tokenize(remove_comments(code))):
    #     print(b)
    #
    # print(f"\n\n\n=== After Output ===")
    # for b in parse(tokenize(remove_comments(code))):
    #     for x in b.output():
    #         print(x)


def make_rand(module):
    # based on https://github.com/russm/lfsr64/blob/master/lfsr64.c
    long_array = ir.ArrayType(long, 256)
    values = [
        0x0000000000000000,
        0xC70000000000000B,
        0x8E0000000000000D,
        0x4900000000000006,
        0x1C00000000000001,
        0xDB0000000000000A,
        0x920000000000000C,
        0x5500000000000007,
        0x3800000000000002,
        0xFF00000000000009,
        0xB60000000000000F,
        0x7100000000000004,
        0x2400000000000003,
        0xE300000000000008,
        0xAA0000000000000E,
        0x6D00000000000005,
        0x7000000000000004,
        0xB70000000000000F,
        0xFE00000000000009,
        0x3900000000000002,
        0x6C00000000000005,
        0xAB0000000000000E,
        0xE200000000000008,
        0x2500000000000003,
        0x4800000000000006,
        0x8F0000000000000D,
        0xC60000000000000B,
        0x0100000000000000,
        0x5400000000000007,
        0x930000000000000C,
        0xDA0000000000000A,
        0x1D00000000000001,
        0xE000000000000008,
        0x2700000000000003,
        0x6E00000000000005,
        0xA90000000000000E,
        0xFC00000000000009,
        0x3B00000000000002,
        0x7200000000000004,
        0xB50000000000000F,
        0xD80000000000000A,
        0x1F00000000000001,
        0x5600000000000007,
        0x910000000000000C,
        0xC40000000000000B,
        0x0300000000000000,
        0x4A00000000000006,
        0x8D0000000000000D,
        0x900000000000000C,
        0x5700000000000007,
        0x1E00000000000001,
        0xD90000000000000A,
        0x8C0000000000000D,
        0x4B00000000000006,
        0x0200000000000000,
        0xC50000000000000B,
        0xA80000000000000E,
        0x6F00000000000005,
        0x2600000000000003,
        0xE100000000000008,
        0xB40000000000000F,
        0x7300000000000004,
        0x3A00000000000002,
        0xFD00000000000009,
        0xC00000000000000B,
        0x0700000000000000,
        0x4E00000000000006,
        0x890000000000000D,
        0xDC0000000000000A,
        0x1B00000000000001,
        0x5200000000000007,
        0x950000000000000C,
        0xF800000000000009,
        0x3F00000000000002,
        0x7600000000000004,
        0xB10000000000000F,
        0xE400000000000008,
        0x2300000000000003,
        0x6A00000000000005,
        0xAD0000000000000E,
        0xB00000000000000F,
        0x7700000000000004,
        0x3E00000000000002,
        0xF900000000000009,
        0xAC0000000000000E,
        0x6B00000000000005,
        0x2200000000000003,
        0xE500000000000008,
        0x880000000000000D,
        0x4F00000000000006,
        0x0600000000000000,
        0xC10000000000000B,
        0x940000000000000C,
        0x5300000000000007,
        0x1A00000000000001,
        0xDD0000000000000A,
        0x2000000000000003,
        0xE700000000000008,
        0xAE0000000000000E,
        0x6900000000000005,
        0x3C00000000000002,
        0xFB00000000000009,
        0xB20000000000000F,
        0x7500000000000004,
        0x1800000000000001,
        0xDF0000000000000A,
        0x960000000000000C,
        0x5100000000000007,
        0x0400000000000000,
        0xC30000000000000B,
        0x8A0000000000000D,
        0x4D00000000000006,
        0x5000000000000007,
        0x970000000000000C,
        0xDE0000000000000A,
        0x1900000000000001,
        0x4C00000000000006,
        0x8B0000000000000D,
        0xC20000000000000B,
        0x0500000000000000,
        0x6800000000000005,
        0xAF0000000000000E,
        0xE600000000000008,
        0x2100000000000003,
        0x7400000000000004,
        0xB30000000000000F,
        0xFA00000000000009,
        0x3D00000000000002,
        0x800000000000000D,
        0x4700000000000006,
        0x0E00000000000000,
        0xC90000000000000B,
        0x9C0000000000000C,
        0x5B00000000000007,
        0x1200000000000001,
        0xD50000000000000A,
        0xB80000000000000F,
        0x7F00000000000004,
        0x3600000000000002,
        0xF100000000000009,
        0xA40000000000000E,
        0x6300000000000005,
        0x2A00000000000003,
        0xED00000000000008,
        0xF000000000000009,
        0x3700000000000002,
        0x7E00000000000004,
        0xB90000000000000F,
        0xEC00000000000008,
        0x2B00000000000003,
        0x6200000000000005,
        0xA50000000000000E,
        0xC80000000000000B,
        0x0F00000000000000,
        0x4600000000000006,
        0x810000000000000D,
        0xD40000000000000A,
        0x1300000000000001,
        0x5A00000000000007,
        0x9D0000000000000C,
        0x6000000000000005,
        0xA70000000000000E,
        0xEE00000000000008,
        0x2900000000000003,
        0x7C00000000000004,
        0xBB0000000000000F,
        0xF200000000000009,
        0x3500000000000002,
        0x5800000000000007,
        0x9F0000000000000C,
        0xD60000000000000A,
        0x1100000000000001,
        0x4400000000000006,
        0x830000000000000D,
        0xCA0000000000000B,
        0x0D00000000000000,
        0x1000000000000001,
        0xD70000000000000A,
        0x9E0000000000000C,
        0x5900000000000007,
        0x0C00000000000000,
        0xCB0000000000000B,
        0x820000000000000D,
        0x4500000000000006,
        0x2800000000000003,
        0xEF00000000000008,
        0xA60000000000000E,
        0x6100000000000005,
        0x3400000000000002,
        0xF300000000000009,
        0xBA0000000000000F,
        0x7D00000000000004,
        0x4000000000000006,
        0x870000000000000D,
        0xCE0000000000000B,
        0x0900000000000000,
        0x5C00000000000007,
        0x9B0000000000000C,
        0xD20000000000000A,
        0x1500000000000001,
        0x7800000000000004,
        0xBF0000000000000F,
        0xF600000000000009,
        0x3100000000000002,
        0x6400000000000005,
        0xA30000000000000E,
        0xEA00000000000008,
        0x2D00000000000003,
        0x3000000000000002,
        0xF700000000000009,
        0xBE0000000000000F,
        0x7900000000000004,
        0x2C00000000000003,
        0xEB00000000000008,
        0xA20000000000000E,
        0x6500000000000005,
        0x0800000000000000,
        0xCF0000000000000B,
        0x860000000000000D,
        0x4100000000000006,
        0x1400000000000001,
        0xD30000000000000A,
        0x9A0000000000000C,
        0x5D00000000000007,
        0xA00000000000000E,
        0x6700000000000005,
        0x2E00000000000003,
        0xE900000000000008,
        0xBC0000000000000F,
        0x7B00000000000004,
        0x3200000000000002,
        0xF500000000000009,
        0x980000000000000C,
        0x5F00000000000007,
        0x1600000000000001,
        0xD10000000000000A,
        0x840000000000000D,
        0x4300000000000006,
        0x0A00000000000000,
        0xCD0000000000000B,
        0xD00000000000000A,
        0x1700000000000001,
        0x5E00000000000007,
        0x990000000000000C,
        0xCC0000000000000B,
        0x0B00000000000000,
        0x4200000000000006,
        0x850000000000000D,
        0xE800000000000008,
        0x2F00000000000003,
        0x6600000000000005,
        0xA10000000000000E,
        0xF400000000000009,
        0x3300000000000002,
        0x7A00000000000004,
        0xBD0000000000000F,
    ]
    feedback_constant = ir.Constant(long_array, values)
    # x =

    # module = ir.Module(name='rand_mod')
    feedback_global = ir.GlobalVariable(module, feedback_constant.type, "rand_feedback")
    feedback_global.linkage = "internal"
    feedback_global.global_constant = True
    feedback_global.initializer = feedback_constant
    fnty = ir.FunctionType(long, (long.as_pointer(), long))
    func = ir.Function(module, fnty, name="rand")

    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    lfsr_pointer, n = func.args
    lfsr = builder.load(lfsr_pointer)
    x = builder.lshr(lfsr, ir.Constant(long, 8))
    feedback_index = builder.and_(lfsr, ir.Constant(long, 0xFF))
    feedback_loc = builder.gep(
        feedback_global.bitcast(ir.ArrayType(long, 0).as_pointer()),
        (ir.Constant(long, 0), feedback_index),
        inbounds=True,
    )
    # feedback_array = builder.load(feedback_array_ptr)
    # feedback_loc = builder.extract_value(feedback_array, (feedback_index, ))
    # feedback_loc = builder.gep(feedback_constant.bitcast(feedback_constant.type.as_pointer()), (ir.Constant(long, 0), feedback_index,))
    # feedback_loc = feedback_constant.bitcast().gep(feedback_index)
    feedback_value = builder.load(feedback_loc)
    new_lfsr = builder.xor(x, feedback_value)
    # builder.store(new_lfsr, lfsr_pointer)
    builder.ret(builder.urem(new_lfsr, n))

    return func  # , module


if "__main__" in __name__:
    main()

# TODO: Make sure can handle: "1 2 3 mget 0 mset +"


class ForfInterpretable(ForfProgram):
    def __init__(self, prog: ForfProg, mem_size: int):
        self._prog = prog
        self._mem_size = mem_size

    def run(self, state: PyForfState):
        if len(state.mem) != self._mem_size:
            raise ValueError("State memory size doesn't match code")
        try:
            self._prog.exec(state)
        except ZeroDivisionError:
            state.error = Error.DIVIDE_BY_ZERO
        except IndexError:
            state.error = Error.OVERFLOW


class ForfExecutable(ForfProgram):
    def __init__(self, cfunc, mem_size: int):
        self._cfunc = cfunc
        self._mem_size = mem_size

    def run(self, state: PyForfState):
        if len(state.mem) != self._mem_size:
            raise ValueError("State memory size doesn't match code")
        res = self._cfunc(pointer(state.mem), pointer(state.rand_seed))
        state.error = Error(res)


class ExecutableCompiler(Compiler):
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
        self._engine = create_execution_engine()

    def compile(self, code: str, mem_size: int = 10) -> ForfProgram:
        from llvmlite import ir

        module = ir.Module(name=__file__)

        func_name = f"func_{uuid.uuid4().hex}"
        ast = parse(tokenize(remove_comments(code)))

        if ForfRand in ast:
            make_rand(module)

        long = ir.IntType(64)
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

    def new_state(self, rand_seed: int) -> ForfState:
        return PyForfState.new(
            func_slots=self._custom_function_set.needed_slots, rand_seed=rand_seed
        )


class InterpretableCompiler(Compiler):
    def compile(self, code: str, mem_size: int = 10) -> ForfProgram:
        prog = parse(tokenize(remove_comments(code)))
        return ForfInterpretable(prog=prog, mem_size=mem_size)

    def new_state(self, rand_seed: int) -> ForfState:
        return PyForfState.new(func_slots=self._custom_function_set.needed_slots)
