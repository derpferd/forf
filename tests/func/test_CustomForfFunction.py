from forf import CustomForfFunction, FunctionInput, FunctionSideEffect, FunctionOutput


def test_happy_path():
    func = CustomForfFunction(
        token="foo",
        inputs=[FunctionInput("a")],
        side_effects=[FunctionSideEffect(12, "b")],
        outputs=[FunctionOutput("c")],
    )
    assert func.get_validation_errors() == []


def test_duplicate_inputs():
    func = CustomForfFunction(
        token="foo", inputs=[FunctionInput("a"), FunctionInput("a")]
    )
    errors = func.get_validation_errors()
    assert len(errors) == 1
    assert "Multiple inputs setting data at 'a'" in errors[0]


def test_duplicate_effects():
    func = CustomForfFunction(
        token="foo",
        side_effects=[FunctionSideEffect(12, "a"), FunctionSideEffect(12, "a")],
    )
    errors = func.get_validation_errors()
    assert len(errors) == 1
    assert "Multiple inputs or side effects setting data at 'a'" in errors[0]


def test_duplicate_input_and_effect():
    func = CustomForfFunction(
        token="foo",
        inputs=[FunctionInput("a")],
        side_effects=[FunctionSideEffect(12, "a")],
    )
    errors = func.get_validation_errors()
    assert len(errors) == 1
    assert "Multiple inputs or side effects setting data at 'a'" in errors[0]


def test_too_many_outputs():
    func = CustomForfFunction(
        token="foo", outputs=[FunctionOutput("a"), FunctionOutput("b")]
    )
    errors = func.get_validation_errors()
    assert len(errors) == 1
    assert (
        "Too many outputs. Got 2 outputs but only support 0 or 1 outputs." in errors[0]
    )


def test_duplicate_input_and_output():
    func = CustomForfFunction(
        token="foo", inputs=[FunctionInput("a")], outputs=[FunctionOutput("a")]
    )
    errors = func.get_validation_errors()
    assert len(errors) == 1
    assert "Outputting data set by input of side effect at 'a'." in errors[0]


def test_duplicate_effect_and_output():
    func = CustomForfFunction(
        token="foo",
        side_effects=[FunctionSideEffect(12, "a")],
        outputs=[FunctionOutput("a")],
    )
    errors = func.get_validation_errors()
    assert len(errors) == 1
    assert "Outputting data set by input of side effect at 'a'." in errors[0]


def test_invalid_token():
    func = CustomForfFunction(token="foo bar")
    errors = func.get_validation_errors()
    assert len(errors) == 1
    assert "Function token can not contain whitespace." in errors[0]
