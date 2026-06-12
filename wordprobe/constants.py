from enum import StrEnum

WORDSIZE = 5
TOKEN_OFFSET = ord("a")
ALPHABET_SIZE = 26
FEEDBACK_CODE_COUNT = 3 ** WORDSIZE
NULL = -1


class TokenFlags(StrEnum):
    FIXED = "-"
    FLOAT = "?"
    MISS = "*"
