"""Pure Python fallback for viterbi_cython Cython extension."""
import numpy as np


def _viterbi_induction(lattice, log_transmatT, framelogprob, n_obs, n_comp, thresh, NINF):
    for t in range(1, n_obs):
        for i in range(n_comp):
            lprb = framelogprob[t, i]
            if lprb < thresh:
                continue
            max_row = NINF
            for j in range(n_comp):
                val = lattice[t - 1, j] + log_transmatT[i, j]
                if val > max_row:
                    max_row = val
            lattice[t, i] = max_row + lprb


def _argmax_row(row, n_comp, NINF):
    """(max_value, argmax) over the first n_comp entries of row."""
    mx = NINF
    max_pos = 0
    for i in range(n_comp):
        if row[i] > mx:
            mx = row[i]
            max_pos = i
    return mx, max_pos


def _viterbi_backtrack(lattice, log_transmatT, state_sequence, n_obs, n_comp, NINF):
    for t in range(n_obs - 2, -1, -1):
        max_row = NINF
        max_pos = 0
        for j in range(n_comp):
            val = lattice[t, j] + log_transmatT[state_sequence[t + 1], j]
            if val > max_row:
                max_row = val
                max_pos = j
        state_sequence[t] = max_pos


def viterbi_cython(n_observations, n_components, log_startprob, log_transmatT, framelogprob):
    NINF = -np.inf
    thresh = np.log(0.0000001)

    viterbi_lattice = NINF * np.ones((n_observations, n_components))
    state_sequence = np.empty(n_observations, dtype=np.intp)

    # Initialization
    for i in range(n_components):
        viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]

    _viterbi_induction(viterbi_lattice, log_transmatT, framelogprob,
                       n_observations, n_components, thresh, NINF)

    # Traceback: pick the best final state, then walk backward.
    _, max_pos = _argmax_row(viterbi_lattice[n_observations - 1], n_components, NINF)
    state_sequence[n_observations - 1] = max_pos
    logprob = float(viterbi_lattice[n_observations - 1, max_pos])

    _viterbi_backtrack(viterbi_lattice, log_transmatT, state_sequence,
                       n_observations, n_components, NINF)

    return logprob, state_sequence
