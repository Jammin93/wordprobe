"""
NumPy-backed scoring, rate, and ranking helpers.

This module is the scoring boundary between the word-oriented engine and the
matrix-oriented heuristic implementation.

Core terminology:
    candidates:
        Current possible answers. These are always the basis for rates.

    pool:
        Words being scored/ranked. If None, pool defaults to candidates.

    scores:
        NumPy array aligned to pool.

Important invariant:
    Rates/weights come from candidates.
    Scores align to pool.
"""
import numpy as np

from .constants import ALPHABET_SIZE, TOKEN_OFFSET, WORDSIZE
from .transforms import tokenize

__all__ = (
    "token_probability_scores",
    "token_entropy_scores",
    "token_index_probability_scores",
    "token_index_entropy_scores",
    "composite_probability_scores",
    "composite_entropy_scores",
    "rank",
    "best_candidate",
    "top_candidates",
)


def _bin_entropy_array(probabilities):
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


def _coerce(candidates, pool=None):
    """
    Convert publically submitted sequences into token matrices. Containing
    ascii letter codes which represent the word tokens (letters).

    Behavior:
        If pool is none, the candidates become the pool.

    Example:
        candidates:
            [ "crane", "corny", "slate" ]

        matrix:
            [
               [ 99,  114, 97,  110, 101 ]
               [ 99,  111, 114, 110, 121 ]
               [ 115, 108, 97,  116, 101 ]
            ]

    This function is the represents the matrix boundary. Public scorers call it
    once and then operate on the returned matrices.
    """
    candidates = tokenize(candidates)
    if pool is None:
        pool = candidates
    else:
        pool = tokenize(pool)

    return candidates, pool


def _token_presence(token_matrix):
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


def _token_rates(token_matrix):
    """
    Compute global token occurrence rates/probabilities from a candidate
    matrix. A token accounts, at most, once per word.

    Example:
        candidates:
            light -> { l, i, g, h, t }
            right -> { r, i, g, h, t }
            night -> { n, i, g, h, t }

        rates:
            i -> 3 / 3 = 1.00
            g -> 3 / 3 = 1.00
            h -> 3 / 3 = 1.00
            t -> 3 / 3 = 1.00
            l -> 1 / 3 = 0.33
            r -> 1 / 3 = 0.33
            n -> 1 / 3 = 0.33
    """
    return _token_presence(token_matrix).mean(axis=0)


def _token_index_rates(token_matrix):
    """
    Compute token-index rates/probabilities from a candidate matrix.

    Example:
        candidates:
            [ "light", "right", "night" ]

        Matrix:
            [
                [ l, i, g, h, t ]
                [ r, i, g, h, t ]
                [ n, i, g, h, t ]
            ]

        Transposed by index:

            index 0: [ l, r, n ]
            index 1: [ i, i, i ]
            index 2: [ g, g, g ]
            index 3: [ h, h, h ]
            index 4: [ t, t, t ]

        Rates:
            rates[i, 1] = 3 / 3 = 1.00
            rates[g, 2] = 3 / 3 = 1.00
            rates[h, 3] = 3 / 3 = 1.00
            rates[t, 4] = 3 / 3 = 1.00
            rates[l, 0] = 1 / 3 = 0.33
            rates[r, 0] = 1 / 3 = 0.33
            rates[n, 0] = 1 / 3 = 0.33

    This is the position-aware counterpart to _token_rates().
    """
    rates = np.zeros((ALPHABET_SIZE, WORDSIZE), dtype=float)
    for i in range(WORDSIZE):
        counts = np.bincount(token_matrix[:, i], minlength=ALPHABET_SIZE)
        rates[:, i] = counts / token_matrix.shape[0]

    return rates


def _score_tokens(matrix, weights, excluded_tokens=None, *, to_average=False):
    """
    Score pool rows using global token rates. The function accepts a token
    matrix and a corresponding wieghting table. Each token in the matrix gets
    multiplied by its average. For binary entropy scoring, this is
    an intermediate step, and we do not average the individual token scores.
    Entropy scores like to be summed; probabilities like to be averaged.

    Example:
        pool:
            [ "trace", "cigar" ]

        presence:
            columns: a c e g i r t
            trace:   1 1 1 0 0 1 1
            cigar:   1 1 0 1 1 1 0

        weights:
            a: .40
            c: .30
            e: .50
            ...

        score:
            Without averaging:
                presence * weight

            Without averaging:
                (presence * weight) / token counts
    """
    excluded_tokens = excluded_tokens or set()
    presence = _token_presence(matrix)
    if excluded_tokens:
        excluded = list(excluded_tokens)
        # If we're excluding a token, we need to multiply by zero, so set its
        # weight in the weights table to zero.
        weights[excluded] = 0.0
        presence[:, excluded] = False

    # multiply weights[x, y] by presence[x, y]
    scores = presence @ weights
    if not to_average:
        return scores

    counts = presence.sum(axis=1)
    return np.divide(
        scores,
        counts,
        out=np.zeros_like(scores, dtype=float),
        where=counts != 0,
    )


def _score_indices(matrix, weights, excluded_indices=None, *, to_average=False):
    """
    Score pool rows using token-index rates.

    Example:
        pool matrix:
            [
                [ c, r, a, n, e ]
                [ s, l, a, t, e ]
            ]

        At index 2:
            tokens = [ a, a ]

        Add:
            rates[a, 2] to both word scores.

    Excluded-index behavior:
        If index 4 is fixed/solved, `excluded_indices` can contain {4}.
        Then no score is assigned for that column.

        This is useful because solved positions no longer provide useful
        positional uncertainty.
    """
    excluded_indices = excluded_indices or set()

    count = 0
    scores = np.zeros(matrix.shape[0], dtype=float)
    for i in range(WORDSIZE):
        if i in excluded_indices:
            continue

        tokens = matrix[:, i]
        scores += weights[tokens, i]
        count += 1

    if to_average and (count > 0):
        scores /= count

    return scores


def _encode_token_set(tokens):
    """Encode a set of tokens into their corresponding ascii ordinal values."""
    if tokens is None:
        return set()

    return {ord(token) - TOKEN_OFFSET for token in tokens}


def token_probability_scores(candidates, pool=None, excluded_tokens=None):
    candidates, pool = _coerce(candidates, pool)
    return _get_token_probability_scores(candidates, pool, excluded_tokens)


def _get_token_probability_scores(candidates, pool=None, excluded_tokens=None):
    """
    Score pool by average global token probability across the list of
    candidates. When no pool is provided, the candidate list becomes the pool.
    """
    rates = _token_rates(candidates)
    excluded_tokens = _encode_token_set(excluded_tokens)
    return _score_tokens(pool, rates, excluded_tokens, to_average=True)


def token_entropy_scores(candidates, pool=None, excluded_tokens=None):
    candidates, pool = _coerce(candidates, pool)
    return _get_token_entropy_scores(candidates, pool, excluded_tokens)


def _get_token_entropy_scores(candidates, pool=None, excluded_tokens=None):
    """
    Score pool by summed global token entropy. When no pool is provided,
    the candidate list becomes the pool.
    """
    entropies = _bin_entropy_array(_token_rates(candidates))
    excluded_tokens = _encode_token_set(excluded_tokens)
    return _score_tokens(pool, entropies, excluded_tokens)


def token_index_probability_scores(
        candidates,
        pool=None,
        excluded_indices=None,
        ):
    candidates, pool = _coerce(candidates, pool)
    return _get_token_index_probability_scores(
        candidates,
        pool,
        excluded_indices,
    )


def _get_token_index_probability_scores(
        candidates,
        pool=None,
        excluded_indices=None,
        ):
    """
    Score pool by average token-index probability. If no pool is provided,
    the pool becomes the candidate list.
    """
    rates = _token_index_rates(candidates)
    return _score_indices(pool, rates, excluded_indices, to_average=True)


def token_index_entropy_scores(candidates, pool=None, excluded_indices=None):
    candidates, pool = _coerce(candidates, pool)
    return _get_token_index_entropy_scores(candidates, pool, excluded_indices)


def _get_token_index_entropy_scores(
        candidates,
        pool=None,
        excluded_indices=None,
        ):
    """
    Score pool by summed token-index entropy. If no pool is provided, the pool
    becomes the candidate list.
    """
    rates = _bin_entropy_array(_token_index_rates(candidates))
    return _score_indices(pool, rates, excluded_indices)


def composite_probability_scores(
        candidates,
        pool=None,
        excluded_tokens=None,
        excluded_indices=None,
        ):
    """
    Score pool by combining token probability and token-index probability. If
    no pool is provided, the pool becomes the candidate list.
    """
    candidates, pool = _coerce(candidates, pool)
    token_scores = _get_token_probability_scores(
        candidates,
        pool,
        excluded_tokens,
    )
    index_scores = _get_token_index_probability_scores(
        candidates,
        pool,
        excluded_indices,
    )
    return (token_scores + index_scores) / 2


def composite_entropy_scores(
        candidates,
        pool=None,
        excluded_tokens=None,
        excluded_indices=None,
        ):
    """
    Score pool by combining token entropy and token-index entropy. If no pool
    is provided, the pool becomes the candidate list.
    """
    candidates, pool = _coerce(candidates, pool)
    token_scores = _get_token_entropy_scores(candidates, pool, excluded_tokens)
    index_scores = _get_token_index_entropy_scores(
        candidates,
        pool,
        excluded_indices,
    )
    return (token_scores + index_scores) / 2


def rank(pool, scores, reverse=True):
    """Rank pool by score."""
    order = np.argsort(scores)
    if reverse:
        order = order[::-1]

    return [
        (pool[int(i)], float(scores[int(i)]))
        for i in order
    ]


def best_candidate(pool, scores, pick_lowest=False):
    """Return the best candidate from the pool, based on scores."""
    pool = np.asarray(pool)
    if pick_lowest:
        i = int(np.argmin(scores))
    else:
        i = int(np.argmax(scores))

    return str(pool[i])


def top_candidates(pool, scores, n=5, pick_lowest=False):
    """Return the top n candidates from the pool, based on scores."""
    return rank(pool, scores, reverse=not pick_lowest)[:n]
