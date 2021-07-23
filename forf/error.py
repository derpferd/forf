from enum import Enum


class Error(Enum):
    NONE = 0
    RUNTIME = 1
    PARSE = 2
    UNDERFLOW = 3
    OVERFLOW = 4
    TYPE = 5
    NO_SUCH_PROCEDURE = 6
    DIVIDE_BY_ZERO = 7
