"""Convert probabilities to binary entropy values."""
import numpy as np

from functools import wraps

from .constants import ALPHABET_SIZE, TOKEN_OFFSET, WORDSIZE
from .transforms import tokenize


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
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    q = 1 - p[mask]

    out[mask] = -((p_mask := p[mask]) * np.log2(p_mask) + q * np.log2(q))
    return out


def _coerce(candidates, pool=None):
    """
    Convert publically submitted sequences into token matrices.    
    
    Behavior:
        If pool is none, the candidates become the pool.

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
    Return a token presence matrix where presence [row, token] is true if it
    appears anywhere in that word. Duplicate tokens in a single word only count
    once. Used for global token rates and token-level scoring.

    Example:
        word tokens:    
                        [ e, e, r, i, e ]
        matrix:         
                        [ e, True ]
                        [ r, True ]
                        [ i, True ]
    """
    presence = np.zeros((token_matrix.shape[0], ALPHABET_SIZE), dtype=bool)
    rows = np.arange(token_matrix.shape[0])[:, None]
    presence[rows, token_matrix] = True
    return presence


def _token_rates(token_matrix):
    """
    Compute global token occurrence rates from a candidate matrix. A token
    accounts, at most, once per word.

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
    rates = np.zeros((ALPHABET_SIZE, WORDSIZE), dtype=float)
    for i in range(WORDSIZE):
        counts = np.bincount(token_matrix[:, i], minlength=ALPHABET_SIZE)
        rates[:, i] = counts / token_matrix.shape[0]

    return rates


def _score_tokens(matrix, rates, excluded_tokens=None, *, to_average=False):
    excluded_tokens = excluded_tokens or set()
    presence = _token_presence(matrix)
    if excluded_tokens:
        excluded = list(excluded_tokens)
        rates[excluded] = 0.0
        presence[excluded] = False

    scores = presence @ rates
    if not to_average:
        return scores

    counts = presence.sum(axis=1)
    return np.divide(
        scores,
        counts,
        out=np.zeros_like(scores, dtype=float),
        where=counts != 0,
    )


def _score_indices(matrix, rates, excluded_indices=None, *, to_average=False):
    excluded_indices = excluded_indices or set()

    count = 0
    scores = np.zeros(matrix.shape[0], dtype=float)
    for i in range(WORDSIZE):
        if i in excluded_indices:
            continue

        tokens = matrix[:, i]
        scores += rates[tokens, i]
        count += 1

    if to_average and (count > 0):
        scores /= count

    return scores


def _encode_token_set(tokens):
    if tokens is None:
        return set()

    return {ord(token) - TOKEN_OFFSET for token in tokens}


def token_probability_scores(candidates, pool=None, excluded_tokens=None):
    candidates, pool = _coerce(candidates, pool)
    rates = _token_rates(candidates)
    excluded_tokens = _encode_token_set(excluded_tokens)
    return _score_tokens(pool, rates, excluded_tokens, to_average=True)


def token_entropy_scores(candidates, pool=None, excluded_tokens=None):
    candidates, pool = _coerce(candidates, pool)
    rates = _bin_entropy_array(_token_rates(candidates))
    excluded_tokens = _encode_token_set(excluded_tokens)
    return _score_tokens(pool, rates, excluded_tokens)


def token_index_probability_scores(
        candidates,
        pool=None,
        excluded_indices=None,
        ):
    candidates, pool = _coerce(candidates, pool)
    rates = _token_index_rates(candidates)
    return _score_indices(pool, rates, excluded_indices, to_average=True)


def token_index_entropy_scores(candidates, pool=None, excluded_indices=None):
    candidates, pool = _coerce(candidates, pool)
    rates = _bin_entropy_array(_token_index_rates(candidates))
    return _score_indices(pool, rates, excluded_indices)


def composite_probability_scores(
        candidates,
        pool=None,
        excluded_tokens=None,
        excluded_indices=None,
        ):
    token_scores = token_probability_scores(candidates, pool, excluded_tokens)
    index_scores = token_index_probability_scores(
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
    token_scores = token_entropy_scores(candidates, pool, excluded_tokens)
    index_scores = token_index_entropy_scores(
        candidates,
        pool,
        excluded_indices,
    )
    return (token_scores + index_scores) / 2


def rank(pool, scores, reverse=True):
    order = np.argsort(scores)
    if reverse:
        order = order[::-1]

    return [
        (pool[int(i)], float(scores[int(i)]))
        for i in order
    ]


def best_candidate(pool, scores, pick_lowest=False):
    pool = np.asarray(pool)
    if pick_lowest:
        i = int(np.argmin(scores))
    else:
        i = int(np.argmax(scores))

    return str(pool[i])


def top_candidates(pool, scores, n=5, pick_lowest=False):
    return rank(pool, scores, reverse=not pick_lowest)[:n]
