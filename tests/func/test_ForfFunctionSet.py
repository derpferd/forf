from forf import (
    ForfFunctionSet,
    CustomForfFunction,
    FunctionInput,
    FunctionSideEffect,
    FunctionOutput,
)


def test_set():
    func_set = ForfFunctionSet(
        (
            CustomForfFunction(
                token="foo",
                inputs=[FunctionInput("a")],
                side_effects=[FunctionSideEffect(12, "b")],
                outputs=[FunctionOutput("c")],
            ),
        )
    )

    funcs = func_set.funcs
    assert len(funcs) == 1
    func = funcs[0]
    assert func.token == "foo"
    assert func.input_slots == [0]
    assert func.effects == [(12, 1)]  # value 12 goes into slot 1
    assert func.output_slots == [2]

    assert func_set.map_slot_to_name(0) == "a"
    assert func_set.map_slot_to_name(1) == "b"
    assert func_set.map_slot_to_name(2) == "c"


# TODO: test edge cases!
