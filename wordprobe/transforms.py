import numpy as np

from collections import Counter

from .constants import ALPHABET_SIZE, TokenFlags, TOKEN_OFFSET, WORDSIZE

# todo: we can cache almost everything produced in this module.


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


def encode_token_set(tokens):
    if tokens is None:
        return set()

    return {ord(token) - TOKEN_OFFSET for token in tokens}


def compose_token_mask(guess, answer):
    mask = [TokenFlags.MISS] * WORDSIZE
    remaining = Counter(answer)
    for i, token in enumerate(guess):
        if token == answer[i]:
            mask[i] = TokenFlags.FIXED
            remaining[token] -= 1

    for i, token in enumerate(guess):
        if mask[i] == TokenFlags.FIXED:
            continue

        if remaining[token] > 0:
            mask[i] = TokenFlags.FLOAT
            remaining[token] -= 1

    return "".join(mask)


def encode_mask(mask):
    code = 0
    fixed = TokenFlags.FIXED
    floating = TokenFlags.FLOAT
    for flag in mask:
        code *= 3
        code += ((flag == floating) * 1) + ((flag == fixed) * 2)

    return code


def decode_mask(code):
    values = "".join((TokenFlags.MISS, TokenFlags.FLOAT, TokenFlags.FIXED))
    flags = [TokenFlags.MISS] * WORDSIZE

    for i in range(WORDSIZE - 1, -1, -1):
        code, value = divmod(int(code), 3)
        flags[i] = values[value]

    return "".join(flags)


def bin_entropy(probabilities):
    """
    Convert an array of probabilies p to an array of binary entropy values.

    Formula:
        E(p) = (-p * log2(p) + (1 - p) * log2(1 - p))

    Boundary Behavior:
        p = 0 -> 0
        p = 1 -> 0

        A probability of 0 or 1 has no uncertainty and is therefore not a good
        candidate for entropy.
    """
    p = np.asarray(probabilities, dtype=float)
    out = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    q = 1 - p[mask]

    out[mask] = -((p_mask := p[mask]) * np.log2(p_mask) + q * np.log2(q))
    return out


def token_presence(token_matrix):
    """
    Return a token presence matrix where presence [row, token] is 1 if it
    appears anywhere in that word, else 0. Duplicate tokens in a single word
    only count once. Used for global token rates and token-level scoring.

    Example:
        word tokens:
            [ e, e, r, i, e ]

        token ordinals:
            [ 101, 101, 114, 105, 101 ]

        return matrix:
            [
                [ 97  (a), False ]
                ...
                [ 101 (e), True  ]
                ...
                [ 105 (i), True  ]
                ...
                [ 114 (r), True  ]
            ]
    """
    presence = np.zeros((token_matrix.shape[0], ALPHABET_SIZE), dtype=np.uint8)
    rows = np.arange(token_matrix.shape[0])[:, None]
    presence[rows, token_matrix] = True
    return presence
