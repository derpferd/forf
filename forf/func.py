"""Classes and code for custom functions"""
import string
from dataclasses import dataclass, field
from typing import List, NamedTuple, Sequence, Dict


class FunctionSetValidationError(Exception):
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Invalid FunctionSet with the following errors: {errors}")


class FunctionInput(NamedTuple):
    dest_data_name: str


class FunctionSideEffect(NamedTuple):
    value: int
    dest_data_name: str


class FunctionOutput(NamedTuple):
    src_data_name: str


@dataclass
class CustomForfFunction:
    # The name of the function
    token: str

    # input order matters, the first input will act on the top of the stack and so on.
    inputs: List[FunctionInput] = field(default_factory=lambda: list())

    side_effects: List[FunctionSideEffect] = field(default_factory=lambda: list())

    # for now we will only support a single output. This must be length 1 or 0
    outputs: List[FunctionOutput] = field(default_factory=lambda: list())

    def get_validation_errors(self) -> List[str]:
        errors = []
        dest_names = set()

        # TODO: more validation for name
        if any(c in self.token for c in string.whitespace):
            errors.append(f"Function token can not contain whitespace.")

        for inp in self.inputs:
            if inp.dest_data_name in dest_names:
                errors.append(f"Multiple inputs setting data at '{inp.dest_data_name}'")
            dest_names.add(inp.dest_data_name)

        for effect in self.side_effects:
            if effect.dest_data_name in dest_names:
                errors.append(
                    f"Multiple inputs or side effects setting data at '{effect.dest_data_name}'"
                )
            dest_names.add(effect.dest_data_name)

        if len(self.outputs) > 1:
            errors.append(
                f"Too many outputs. Got {len(self.outputs)} outputs but only support 0 or 1 outputs."
            )
        elif len(self.outputs) == 1:
            output = self.outputs[0]
            if output.src_data_name in dest_names:
                errors.append(
                    f"Outputting data set by input of side effect at '{output.src_data_name}'."
                )

        return errors


class ForfFunctionSet:
    class CompiledSideEffect(NamedTuple):
        value: int
        slot: int

    class CompiledFunction(NamedTuple):
        token: str
        input_slots: List[int]
        effects: List["ForfFunctionSet.CompiledSideEffect"]
        output_slots: List[int]

    def __init__(self, functions: Sequence[CustomForfFunction]):
        self._functions = functions
        self._func_table = {}
        self._data_name_to_slot: Dict[str, int] = {}
        self._slot_to_data_name: Dict[int, str] = {}

        self._compile()

    @property
    def funcs(self) -> Sequence[CompiledFunction]:
        return list(self._func_table.values())

    @property
    def needed_slots(self) -> int:
        return len(self._data_name_to_slot)

    def map_slot_to_name(self, slot: int) -> str:
        return self._slot_to_data_name[slot]

    def map_name_to_slot(self, name: str) -> int:
        return self._data_name_to_slot[name]

    def _get_validation_errors(self) -> List[str]:
        errors = []
        func_tokens = set()
        for func in self._functions:
            if func.token in func_tokens:
                errors.append(
                    f"At least two functions have the token '{func.token}'. Each function must have a unique token."
                )
            func_tokens.add(func.token)
            func_errors = func.get_validation_errors()
            if func_errors:
                errors.extend(
                    [f"Function '{func.token}' has error: {err}" for err in func_errors]
                )
        return errors

    def _compile(self):
        self._func_table = {}
        self._data_name_to_slot = {}
        self._slot_to_data_name = {}

        errors = self._get_validation_errors()
        if errors:
            raise FunctionSetValidationError(errors)

        for func in self._functions:
            input_slots = []
            for inp in func.inputs:
                if inp.dest_data_name not in self._data_name_to_slot:
                    new_slot = len(self._data_name_to_slot)
                    self._data_name_to_slot[inp.dest_data_name] = new_slot
                    self._slot_to_data_name[new_slot] = inp.dest_data_name
                input_slots.append(self._data_name_to_slot[inp.dest_data_name])

            effects = []
            for effect in func.side_effects:
                if effect.dest_data_name not in self._data_name_to_slot:
                    new_slot = len(self._data_name_to_slot)
                    self._data_name_to_slot[effect.dest_data_name] = new_slot
                    self._slot_to_data_name[new_slot] = effect.dest_data_name
                effects.append(
                    self.CompiledSideEffect(
                        effect.value, self._data_name_to_slot[effect.dest_data_name]
                    )
                )

            output_slots = []
            for out in func.outputs:
                if out.src_data_name not in self._data_name_to_slot:
                    new_slot = len(self._data_name_to_slot)
                    self._data_name_to_slot[out.src_data_name] = new_slot
                    self._slot_to_data_name[new_slot] = out.src_data_name
                output_slots.append(self._data_name_to_slot[out.src_data_name])

            self._func_table[func.token] = self.CompiledFunction(
                func.token,
                input_slots=input_slots,
                effects=effects,
                output_slots=output_slots,
            )
