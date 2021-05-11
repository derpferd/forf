import pytest
from forf.base import *


def test_tokenize():
    # TODO: support non whitespace
    s = "5 8\t< { ifelse\ndup\t \ndup"
    assert list(tokenize(s)) == ["5", "8", "<", "{", "ifelse", "dup", "dup"]


def test_remove_comments():
    s = """get-turret 12 + set-turret!         ( Rotate turret )
37 40 set-speed!                    ( Go in circles )
0 sensor? { fire! } if              ( Fire if turret sensor triggered )
1 sensor? { -50 50 set-speed! } if  ( Turn if collision sensor triggered )
1 9 mset  ( adsf ( dsaf )
0 mget (HI) set-led!
6 1 + (asd
afdsa )"""
    assert (
        remove_comments(s)
        == """get-turret 12 + set-turret!         
37 40 set-speed!                    
0 sensor? { fire! } if              
1 sensor? { -50 50 set-speed! } if  
1 9 mset  
0 mget  set-led!
6 1 + """
    )


B = ForfProg.Block


@pytest.mark.parametrize(
    "tokens, output",
    (
        (
            ["58", "58", "*"],
            ForfProg([ForfProg.Block(ForfBin("*", [ForfValue(58), ForfValue(58)]))]),
        ),
        (
            ["58", "58", "58", "*", "*"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfBin(
                            "*",
                            [
                                ForfValue(58),
                                ForfBin("*", [ForfValue(58), ForfValue(58)]),
                            ],
                        )
                    )
                ]
            ),
        ),
        (
            ["58", "dup", "dup", "*", "*"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfBin(
                            "*",
                            [
                                ForfValue(58),
                                ForfBin("*", [ForfValue(58), ForfValue(58)]),
                            ],
                        )
                    )
                ]
            ),
        ),
        (
            ["1", "2", ">", "{", "3", "}", "if"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "if",
                            ForfBin(">", [ForfValue(1), ForfValue(2)]),
                            [ForfProg.Block(ForfValue(3))],
                            [],
                        )
                    )
                ]
            ),
        ),
        (
            ["1", "2", ">", "{", "3", "}", "{", "4", "}", "ifelse"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "ifelse",
                            ForfBin(">", [ForfValue(1), ForfValue(2)]),
                            [B(ForfValue(3))],
                            [B(ForfValue(4))],
                        )
                    )
                ]
            ),
        ),
        (
            ["1", "2", "3", ">", "{", "4", "+", "}", "if"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "if",
                            ForfBin(">", [ForfValue(2), ForfValue(3)]),
                            [B(ForfBin("+", [ForfValue(1), ForfValue(4)]))],
                            [B(ForfValue(1))],
                        )
                    )
                ]
            ),
        ),
        (
            [
                "1",
                "2",
                "3",
                ">",
                "{",
                "4",
                "+",
                "}",
                "{",
                "5",
                "6",
                "<",
                "{",
                "7",
                "+",
                "}",
                "if",
                "}",
                "ifelse",
            ],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "ifelse",
                            ForfBin(">", [ForfValue(2), ForfValue(3)]),
                            [B(ForfBin("+", [ForfValue(1), ForfValue(4)]))],
                            [
                                B(
                                    ForfIfElse(
                                        "if",
                                        ForfBin("<", [ForfValue(5), ForfValue(6)]),
                                        [B(ForfBin("+", [ForfValue(1), ForfValue(7)]))],
                                        [B(ForfValue(1))],
                                    )
                                )
                            ],
                        )
                    )
                ]
            ),
        ),
        (
            [
                "5",
                "8",
                "<",
                "{",
                "50",
                "8",
                "+",
                "}",
                "{",
                "50",
                "8",
                "-",
                "}",
                "ifelse",
            ],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "ifelse",
                            ForfBin("<", [ForfValue(5), ForfValue(8)]),
                            [B(ForfBin("+", [ForfValue(50), ForfValue(8)]))],
                            [B(ForfBin("-", [ForfValue(50), ForfValue(8)]))],
                        )
                    )
                ]
            ),
        ),
        (
            ["50", "5", "8", "<", "{", "8", "+", "}", "{", "8", "-", "}", "ifelse"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "ifelse",
                            ForfBin("<", [ForfValue(5), ForfValue(8)]),
                            [B(ForfBin("+", [ForfValue(50), ForfValue(8)]))],
                            [B(ForfBin("-", [ForfValue(50), ForfValue(8)]))],
                        )
                    )
                ]
            ),
        ),
        (
            ["50", "8", "5", "8", "<", "{", "+", "}", "{", "-", "}", "ifelse"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "ifelse",
                            ForfBin("<", [ForfValue(5), ForfValue(8)]),
                            [B(ForfBin("+", [ForfValue(50), ForfValue(8)]))],
                            [B(ForfBin("-", [ForfValue(50), ForfValue(8)]))],
                        )
                    )
                ]
            ),
        ),
        (
            ["3", "mget", "100", ">"],
            ForfProg(
                [
                    B(ForfMget("mget", [ForfValue(3)]), assigned_var=0, used=True),
                    B(
                        ForfBin(
                            ">",
                            [
                                ForfVar(0),
                                ForfValue(100),
                            ],
                        )
                    ),
                ],
                num_vars=1,
            ),
        ),
        (
            [
                "4",
                "dup",
                "dup",
                "*",
                "100",
                ">",
                "{",
                "dup",
                "*",
                "}",
                "{",
                "pop",
                "0",
                "}",
                "ifelse",
            ],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "ifelse",
                            ForfBin(
                                ">",
                                [
                                    ForfBin("*", [ForfValue(4), ForfValue(4)]),
                                    ForfValue(100),
                                ],
                            ),
                            [
                                B(ForfBin("*", [ForfValue(4), ForfValue(4)])),
                            ],
                            [B(ForfValue(0))],
                        )
                    )
                ]
            ),
        ),
        (
            ["4", "dup", "*", "dup", "100", "<", "{", "pop", "0", "}", "if"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "if",
                            ForfBin(
                                "<",
                                [
                                    ForfBin("*", [ForfValue(4), ForfValue(4)]),
                                    ForfValue(100),
                                ],
                            ),
                            [B(ForfValue(0))],
                            [
                                B(ForfBin("*", [ForfValue(4), ForfValue(4)])),
                            ],
                        )
                    )
                ]
            ),
        ),
        (
            ["1", "2", "dup", "*", "exch", "dup", "*", "+", "88", "88", "*", "<"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfBin(
                            "<",
                            [
                                ForfBin(
                                    "+",
                                    [
                                        ForfBin("*", [ForfValue(2), ForfValue(2)]),
                                        ForfBin("*", [ForfValue(1), ForfValue(1)]),
                                    ],
                                ),
                                ForfBin("*", [ForfValue(88), ForfValue(88)]),
                            ],
                        )
                    )
                ]
            ),
        ),
        (
            ["1", "2", ">", "{", "3", "4", "}", "{", "5", "6", "}", "ifelse", "+"],
            ForfProg(
                [
                    ForfProg.Block(
                        ForfIfElse(
                            "ifelse",
                            ForfBin(">", [ForfValue(1), ForfValue(2)]),
                            [B(ForfBin("+", [ForfValue(3), ForfValue(4)]))],
                            [B(ForfBin("+", [ForfValue(5), ForfValue(6)]))],
                        )
                    )
                ]
            ),
        ),
    ),
)
def test_parse(tokens, output):
    assert parse(tokens) == output


# def test_parse_rand():
#     assert parse(['1', 'random']) == [ForfRand('random', [ForfValue(1)])]


# @pytest.mark.parametrize('tokens', (
#     ['1', '2', '>', '{', '3', '}', '{', '4', '5', '6', '7', '}', 'ifelse', '+'],
# ))
# def test_parse_pop_from_empty_list(tokens):
#     with pytest.raises(IndexError) as excinfo:
#         parse(tokens)
#     assert "pop from empty stack" in str(excinfo.value)


def test_exec():
    state = exec("1 2 + 0 mset 3 4 + 1 mset")
    assert state.mem[0] == 3
    assert state.mem[1] == 7

    assert exec("58 58 * 0 mset").mem[0] == 58 ** 2
    assert exec("58 dup dup * * 0 mset").mem[0] == 58 ** 3
    assert exec("5 8 < { 50 8 + } { 50 8 - } ifelse 0 mset").mem[0] == 58


# @pytest.mark.parametrize('num,in_list,inputs,out_list', (
#     (1, [ForfValue(1)], [ForfValue(1)], []),
#     (2, [ForfValue(1), ForfValue(2)], [ForfValue(1), ForfValue(2)], []),
#     (2, [ForfValue(1), ForfMset('mset'), ForfValue(2)], [ForfValue(1), ForfValue(2)], [ForfMset('mset')]),
# ))
# def test_get_inputs_from_list(num, in_list, inputs, out_list):
#     assert _get_inputs_from_list(num, in_list) == inputs
#     assert in_list == out_list
