import numpy as np

from .constants import TOKEN_OFFSET, WORDSIZE


def encode_tokens(word):
    return np.fromiter(
        [ord(token) - TOKEN_OFFSET for token in word],
        dtype=np.uint8,
        count=WORDSIZE,
    )


def decode_tokens(word):
    return np.fromiter([int(token) + TOKEN_OFFSET for token in word], np.uint8)


def tokenize(words):
    if not words:
        return np.empty((0, WORDSIZE), dtype=np.uint8)

    matrix = np.empty((len(words), WORDSIZE), dtype=np.uint8)
    for row, word in enumerate(words):
        for col, v in enumerate(encode_tokens(word)):
            matrix[row, col] = v

    return matrix


def compose(matrix):
    return np.array(decode_tokens(row) for row in matrix)
