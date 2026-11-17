"""Independent parsers: derive every target from prompt tokens, never source metadata."""

import numpy as np


def ruler_oracle(prompt, config):
    """Build RHS -> LHS edges and follow the queried root through every alias."""
    prompt = list(map(int, prompt))
    if len(prompt) != config.prompt_length or prompt[-2] != 3:
        raise ValueError("invalid RULER query/length")
    edges, left, roots = {}, [], []
    cursor, filler = 0, 0
    while cursor < len(prompt) - 2:
        token = prompt[cursor]
        if 4 <= token < 20:
            filler += 1
            cursor += 1
            continue
        statement = prompt[cursor : cursor + 4]
        if len(statement) != 4:
            raise ValueError("truncated RULER assignment")
        lhs, equals, rhs, separator = statement
        if not (20 <= lhs < 20 + config.n_vars and equals == 1 and separator == 2):
            raise ValueError("invalid RULER assignment")
        if lhs in left or rhs in edges:
            raise ValueError("RULER names/edges must be unique")
        if 20 + config.n_vars <= rhs < config.vocab_size:
            roots.append(rhs)
        elif rhs not in left:
            raise ValueError("RULER alias precedes its assignment")
        edges[rhs] = lhs
        left.append(lhs)
        cursor += 4
    if filler != config.n_filler or len(roots) != config.num_chains:
        raise ValueError("RULER filler/root count mismatch")
    if len(left) != config.num_chains * config.answer_length or len(set(roots)) != len(roots):
        raise ValueError("RULER chain count mismatch")
    chains = []
    for root in roots:
        chain, node = [], root
        while node in edges:
            node = edges[node]
            chain.append(node)
            if len(chain) > config.answer_length:
                raise ValueError("RULER cycle/length mismatch")
        if len(chain) != config.answer_length:
            raise ValueError("RULER chain length mismatch")
        chains.append(chain)
    if prompt[-1] not in roots:
        raise ValueError("RULER query root missing")
    return np.asarray(chains[roots.index(prompt[-1])], dtype=np.int64), {
        "edges": edges,
        "blind_lhs": left[: config.answer_length],
    }


def state_oracle(prompt, config):
    """Apply swaps to the parsed pointer map; moved means ADDRESS, not bit changed."""
    p = list(map(int, prompt))
    n = config.n_bits
    if len(p) != config.prompt_length or any(bit not in (4, 5) for bit in p[:n]):
        raise ValueError("invalid state recall bits/length")
    if p[n] != 1 or p[n + 11] != 2 or p[-2:] != [3, 6]:
        raise ValueError("invalid state recall markers/query")
    if p[n + 1 : n + 11 : 2] != list(range(6, 11)):
        raise ValueError("invalid state recall pointer names")
    initial = [value - 11 for value in p[n + 2 : n + 11 : 2]]
    if len(set(initial)) != 5 or any(not 0 <= address < n for address in initial):
        raise ValueError("invalid state recall addresses")
    pointers = initial.copy()
    for index in range(n + 12, len(p) - 2, 2):
        a, b = p[index] - 6, p[index + 1] - 6
        if not (0 <= a < 5 and 0 <= b < 5 and a != b):
            raise ValueError("invalid state recall swap")
        pointers[a], pointers[b] = pointers[b], pointers[a]
    answer = p[pointers[0]]
    return np.asarray([answer], dtype=np.int64), {
        "address": pointers[0],
        "initial_address": initial[0],
        "address_moved": pointers[0] != initial[0],
        "bit_changed": answer != p[initial[0]],
        "ignore_swaps": p[initial[0]],
    }
