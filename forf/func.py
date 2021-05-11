"""Classes and code for custom functions"""
from dataclasses import dataclass, field
from typing import List, NamedTuple


class FunctionInput(NamedTuple):
    dest_data_name: str


class FunctionSideEffect(NamedTuple):
    value: int
    dest_data_name: str


class FunctionOutput(NamedTuple):
    src_data_name: str


@dataclass
class CustomForfFunction:
    # input order matters, the first input will act on the top of the stack and so on.
    inputs: List[FunctionInput] = field(default_factory=lambda: list())

    side_effects: List[FunctionSideEffect] = field(default_factory=lambda: list())

    # for now we will only support a single output. This must be length 1 or 0
    outputs: List[FunctionOutput] = field(default_factory=lambda: list())

    def get_validation_errors(self) -> List[str]:
        errors = []
        dest_names = set()

        for inp in self.inputs:
            if inp.dest_data_name in dest_names:
                errors.append(f"Multiple inputs setting data at '{inp.dest_data_name}'")
            dest_names.add(inp.dest_data_name)

        for effect in self.side_effects:
            if effect.dest_data_name in dest_names:
                errors.append(f"Multiple inputs or side effects setting data at '{effect.dest_data_name}'")
            dest_names.add(effect.dest_data_name)

        if len(self.outputs) > 1:
            errors.append(f"Too many outputs. Got {len(self.outputs)} outputs but only support 0 or 1 outputs.")
        elif len(self.outputs) == 1:
            output = self.outputs[0]
            if output.src_data_name in dest_names:
                errors.append(f"Outputting data set by input of side effect at '{output.src_data_name}'.")

        return errors
