from functools import lru_cache
from itertools import combinations

import numpy as np
from numba import njit
from numba import types
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.unsafe.ndarray import to_fixed_tuple
from numba.typed import List, Dict


def sort_filtration_by_dim(filtration, maxdim=None):
    if maxdim is None:
        maxdim = max(map(len, filtration)) - 1

    filtration_by_dim = [[] for _ in range(maxdim + 1)]
    for idx, v in enumerate(filtration):
        v_t = tuple(sorted(v))
        dim = len(v_t) - 1
        if dim <= maxdim:
            filtration_by_dim[dim].append([idx, v_t])
    
    for dim, filtr in enumerate(filtration_by_dim):
        filtration_by_dim[dim] = [np.asarray(x, dtype=np.int64)
                                  for x in zip(*filtr)]

    return filtration_by_dim


@njit
def _twist_reduction(coboundary, triangular, pivots_lookup, idxs_next_dim):
    """R = MV"""
    n = len(coboundary)

    pos_idxs_to_clear = List.empty_list(types.int64)
    for j in range(n):
        highest_one = coboundary[j][0] if coboundary[j] else -1
        pivot_col = pivots_lookup[highest_one]
        while (highest_one != -1) and (pivot_col != -1):
            coboundary[j] = _symm_diff(coboundary[j][1:],
                                       coboundary[pivot_col][1:])
            triangular[j] = _symm_diff(triangular[j],
                                       triangular[pivot_col])
            highest_one = coboundary[j][0] if coboundary[j] else -1
            pivot_col = pivots_lookup[highest_one]
        if highest_one != -1:
            pivots_lookup[highest_one] = j
            pos_idxs_to_clear.append(highest_one)

    for j in range(n):
        coboundary[j] = [idxs_next_dim[k] for k in coboundary[j]]

    return coboundary, triangular, pos_idxs_to_clear


@lru_cache
def _reduce_single_dim(dim):
    len_tups_dim = dim + 1
    tuple_typ_dim = types.UniTuple(types.int64, len_tups_dim)
    len_tups_next_dim = dim + 2
    int64_list_typ = types.List(types.int64)

    @njit
    def _inner_reduce_single_dim(idxs_dim, tups_dim, pos_idxs_to_clear,
                                 idxs_next_dim=None, tups_next_dim=None):
        """R = MV"""
        spx2idx_dim = Dict.empty(tuple_typ_dim, types.int64)
        # Initialize reduced matrix as the coboundary matrix
        reduced = List.empty_list(int64_list_typ)
        triangular = List.empty_list(int64_list_typ)
        for i in range(len(idxs_dim) - 1, -1, -1):
            spx = to_fixed_tuple(tups_dim[i], len_tups_dim)
            spx2idx_dim[spx] = i
            reduced.append([types.int64(x) for x in range(0)])
            triangular.append([idxs_dim[i]])

        if idxs_next_dim is not None:
            for j in range(len(idxs_next_dim)):
                spx = to_fixed_tuple(tups_next_dim[j], len_tups_next_dim)
                for face in _drop_elements(spx):
                    reduced[-1 - spx2idx_dim[face]].append(j)

            for pos_idx in pos_idxs_to_clear:
                reduced[-1 - pos_idx] = [types.int64(x) for x in range(0)]

            pivots_lookup = [-1] * len(idxs_next_dim)

            reduced, triangular, pos_idxs_to_clear = _twist_reduction(
                reduced, triangular, pivots_lookup, idxs_next_dim
                )

        return spx2idx_dim, reduced, triangular, pos_idxs_to_clear

    return _inner_reduce_single_dim


def get_reduced_triangular(filtr_by_dim):
    maxdim = len(filtr_by_dim) - 1
    spx2idx_idxs_reduced_triangular = []
    pos_idxs_to_clear = List.empty_list(types.int64)
    for dim in range(maxdim):
        reduction_dim = _reduce_single_dim(dim)
        idxs_dim, tups_dim = filtr_by_dim[dim]
        idxs_next_dim, tups_next_dim = filtr_by_dim[dim + 1]
        spx2idx_dim, reduced, triangular, pos_idxs_to_clear = reduction_dim(
            idxs_dim,
            tups_dim,
            pos_idxs_to_clear,
            idxs_next_dim=idxs_next_dim,
            tups_next_dim=tups_next_dim
            )
        spx2idx_idxs_reduced_triangular.append((spx2idx_dim,
                                                idxs_dim[::-1],
                                                reduced,
                                                triangular))

    reduction_dim = _reduce_single_dim(maxdim)
    idxs_dim, tups_dim = filtr_by_dim[maxdim]
    spx2idx_dim, reduced, triangular, _ = reduction_dim(
        idxs_dim,
        tups_dim,
        os_idxs_to_clear
        )
    spx2idx_idxs_reduced_triangular.append((spx2idx_dim,
                                            idxs_dim[::-1],
                                            reduced,
                                            triangular))

    return spx2idx_idxs_reduced_triangular


def get_barcode(N, spx2idx_idxs_reduced_triangular,
                filtration_values=None):
    def is_nontrivial_bar(b, d):
        return filtration_values[N - 1 - b] != filtration_values[N - 1 - d]

    pairs = []
    all_indices = set()

    if filtration_values is None:
        _, idxs_0, reduced_0, _ = spx2idx_idxs_reduced_triangular[0]
        pairs_0 = []
        for i in range(len(idxs_0)):
            if not reduced_0[i]:
                pairs_0.append((N - 1 - idxs_0[i], np.inf))
        pairs.append(sorted(pairs_0))

        for dim in range(1, len(spx2idx_idxs_reduced_triangular)):
            _, idxs_dim, reduced_dim, _ = spx2idx_idxs_reduced_triangular[dim]
            _, idxs_prev_dim, reduced_prev_dim, _ = \
                spx2idx_idxs_reduced_triangular[dim - 1]

            pairs_dim = []
            for i in range(len(idxs_prev_dim)):
                if reduced_prev_dim[i]:
                    b = N - 1 - reduced_prev_dim[i][0]
                    d = N - 1 - idxs_prev_dim[i]
                    pairs_dim.append((b, d))
                    all_indices |= {b, d}

            for i in range(len(idxs_dim)):
                if N - 1 - idxs_dim[i] not in all_indices:
                    if not reduced_dim[i]:
                        pairs_dim.append((N - 1 - idxs_dim[i], np.inf))

            pairs.append(sorted(pairs_dim))

    else:
        _, idxs_0, reduced_0, _ = spx2idx_idxs_reduced_triangular[0]
        pairs_0 = []
        for i in range(len(idxs_0)):
            if not reduced_0[i]:
                pairs_0.append((N - 1 - idxs_0[i], np.inf))
        pairs.append(sorted(pairs_0))

        for dim in range(1, len(spx2idx_idxs_reduced_triangular)):
            _, idxs_dim, reduced_dim, _ = spx2idx_idxs_reduced_triangular[dim]
            _, idxs_prev_dim, reduced_prev_dim, _ = \
                spx2idx_idxs_reduced_triangular[dim - 1]

            pairs_dim = []
            for i in range(len(idxs_prev_dim)):
                if reduced_prev_dim[i]:
                    b = N - 1 - reduced_prev_dim[i][0]
                    d = N - 1 - idxs_prev_dim[i]
                    if is_nontrivial_bar(b, d):
                        pairs_dim.append((b, d))
                    all_indices |= {b, d}

            for i in range(len(idxs_dim)):
                if N - 1 - idxs_dim[i] not in all_indices:
                    if not reduced_dim[i]:
                        pairs_dim.append((N - 1 - idxs_dim[i], np.inf))

            pairs.append(sorted(pairs_dim))

    return pairs


def get_coho_reps(N, barcode, spx2idx_idxs_reduced_triangular):
    coho_reps = []
    
    _, idxs_0, _, triangular_0 = spx2idx_idxs_reduced_triangular[0]
    coho_reps_0 = []
    for pair in barcode[0]:
        idx = np.flatnonzero(idxs_0 == N - 1 - pair[0])[0]
        coho_reps_0.append([N - 1 - x for x in triangular_0[idx]])
    coho_reps.append(coho_reps_0)
    
    for dim in range(1, len(barcode)):
        barcode_dim = barcode[dim]
        _, idxs_dim, _, triangular_dim = spx2idx_idxs_reduced_triangular[dim]
        _, idxs_prev_dim, reduced_prev_dim, _ = \
            spx2idx_idxs_reduced_triangular[dim - 1]

        coho_reps_dim = []
        for pair in barcode_dim:
            if pair[1] < np.inf:
                idx = np.flatnonzero(idxs_prev_dim == N - 1 - pair[1])[0]
                coho_reps_dim.append([N - 1 - x
                                      for x in reduced_prev_dim[idx]])
            else:
                idx = np.flatnonzero(idxs_dim == N - 1 - pair[0])[0]
                coho_reps_dim.append([N - 1 - x for x in triangular_dim[idx]])
                
        coho_reps.append(coho_reps_dim)

    return coho_reps


@njit
def _tuple_in_dict(tup, d):
    return tup in d
    

def STSQ(length, cocycle, filtration):
    """..."""
    answer = set()
    for pair in combinations(cocycle, 2):
        a, b = set(pair[0]), set(pair[1])
        u = a.union(b)
        if len(u) == length:
            u_tuple = tuple(sorted(u))
            if _tuple_in_dict(u_tuple, filtration):
                a_bar, b_bar = a.difference(b), b.difference(a)
                u_bar = sorted(a_bar.union(b_bar))
                index = {}
                for v in a_bar.union(b_bar):
                    pos = u_tuple.index(v)
                    pos_bar = u_bar.index(v)
                    index[v] = (pos + pos_bar) % 2
                index_a = {index[v] for v in a_bar}
                index_b = {index[w] for w in b_bar}
                if (index_a == {0} and index_b == {1}
                        or index_a == {1} and index_b == {0}):
                    answer ^= {u_tuple}

    return answer


@njit
def _populate_st_mat(st_mat, cochain, idxs, spx2idx_dim):
    st_mat.append(sorted([idxs[-1 - spx2idx_dim[spx]] for spx in cochain]))


@njit
def _populate_st_mat_with_empty(st_mat):
    st_mat.append([types.int64(x) for x in range(0)])


def get_steenrod_matrix(k, coho_reps, filtration,
                        spx2idx_idxs_reduced_triangular):
    steenrod_matrix = [list()] * k
    for dim, coho_reps_dim in enumerate(coho_reps[:-1]):
        length = dim + 1 + k
        steenrod_matrix.append(List.empty_list(types.List(types.int64)))
        for i, rep in enumerate(coho_reps_dim):
            cocycle = [filtration[-1 - j] for j in rep]
            spx2idx_dim_plus_k, idxs_dim_plus_k, _, _ = \
                spx2idx_idxs_reduced_triangular[dim + k]
            cochain = STSQ(length, cocycle, spx2idx_dim_plus_k)
            if cochain:
                _populate_st_mat(steenrod_matrix[dim + k],
                                 cochain,
                                 idxs_dim_plus_k,
                                 spx2idx_dim_plus_k)
            else:
                _populate_st_mat_with_empty(steenrod_matrix[dim + k])
        
    return steenrod_matrix


@njit
def _steenrod_barcode_single_dim(steenrod_matrix_dim, idxs_prev_dim,
                                 reduced_prev_dim, births_dim, N):
    augmented = []
    for i in range(len(reduced_prev_dim) - 1, -1, -1):
        augmented.append([x for x in reduced_prev_dim[i]])

    for i in range(len(steenrod_matrix_dim)):
        augmented.append([x for x in steenrod_matrix_dim[i]])

    alive = [True] * len(births_dim)
    n = len(reduced_prev_dim)
    st_barcode_dim = []

    j = 0
    for i, idx in enumerate(idxs_prev_dim):
        if births_dim[j] == idx:
            j += 1
        for ii in range(n, n + j):
            if augmented[ii]:
                iii = ii
                while iii >= n - i:
                    iii -= 1
                    if not augmented[ii]:
                        break
                    elif not augmented[iii]:
                        continue
                    elif augmented[iii][0] == augmented[ii][0]:
                        augmented[ii] = _symm_diff(augmented[iii][1:],
                                                   augmented[ii][1:])
                        iii = ii

            if alive[ii - n] and (not augmented[ii]):
                alive[ii - n] = False
                if idx < births_dim[ii - n]:
                    st_barcode_dim.append((N - 1 - births_dim[ii - n],
                                           N - 1 - idx))

    for i in range(len(alive)):
        if alive[i]:
            st_barcode_dim.append((N - 1 - births_dim[i], -1))

    return st_barcode_dim


def get_steenrod_barcode(k, steenrod_matrix, spx2idx_idxs_reduced_triangular,
                         barcode, N, filtration_values=None):
    def is_nontrivial_bar(pair):
        return (pair[1] != -1) and (filtration_values[N - 1 - pair[0]] !=
                                    filtration_values[N - 1 - pair[1]])

    st_barcode = [list()] * k
    if filtration_values is None:
        for dim in range(k, len(steenrod_matrix)):
            if steenrod_matrix[dim]:
                births_dim = \
                    np.asarray([N - 1 - pair[0] for pair in barcode[dim - k]],
                               dtype=np.int64)
                _, idxs_prev_dim, reduced_prev_dim, _ = \
                    spx2idx_idxs_reduced_triangular[dim - 1]
                st_barcode.append([
                    pair if pair[1] != -1 else (pair[0], np.inf)
                    for pair in _steenrod_barcode_single_dim(
                        steenrod_matrix[dim],
                        idxs_prev_dim,
                        reduced_prev_dim,
                        births_dim,
                        N
                        )
                    ])
            else:
                st_barcode.append([])

    else:
        for dim in range(k, len(steenrod_matrix)):
            if steenrod_matrix[dim]:
                births_dim = \
                    np.asarray([N - 1 - pair[0] for pair in barcode[dim - k]],
                               dtype=np.int64)
                _, idxs_prev_dim, reduced_prev_dim, _ = \
                    spx2idx_idxs_reduced_triangular[dim - 1]
                st_barcode.append([
                    pair if is_nontrivial_bar(pair) else (pair[0], np.inf)
                    for pair in _steenrod_barcode_single_dim(
                        steenrod_matrix[dim],
                        idxs_prev_dim,
                        reduced_prev_dim,
                        births_dim,
                        N
                        )
                    ])
            else:
                st_barcode.append([])

    return st_barcode
                        

def barcodes(
        k, filtration, homology=False, filtration_values=None,
        return_filtration_values=False, maxdim=None
        ):
    """Serves as the main function"""
    N = len(filtration)
    filtration_by_dim = sort_filtration_by_dim(filtration, maxdim=maxdim)
    spx2idx_idxs_reduced_triangular = get_reduced_triangular(filtration_by_dim)
    barcode = get_barcode(N, spx2idx_idxs_reduced_triangular,
                          filtration_values=filtration_values)
    coho_reps = get_coho_reps(N, barcode, spx2idx_idxs_reduced_triangular)
    steenrod_matrix = get_steenrod_matrix(k, coho_reps, filtration,
                                          spx2idx_idxs_reduced_triangular)
    st_barcode = get_steenrod_barcode(k, steenrod_matrix,
                                      spx2idx_idxs_reduced_triangular, barcode,
                                      N, filtration_values=filtration_values)

    if homology:
        barcode = to_homology_barcode(
            barcode, N, filtration_values=filtration_values,
            return_filtration_values=return_filtration_values
            )
        st_barcode = to_homology_barcode(
            st_barcode, N, filtration_values=filtration_values,
            return_filtration_values=return_filtration_values
            )

        return barcode, st_barcode

    if return_filtration_values and (filtration_values is not None):
        barcode = to_values_barcode(barcode, N, filtration_values)
        st_barcode = to_values_barcode(st_barcode, N, filtration_values)

    return barcode, st_barcode


def to_homology_barcode(rel_coho_barcode, N, filtration_values=None,
                        return_filtration_values=True):
    hom_barcode = []

    if (not return_filtration_values) or (filtration_values is None):
        for dim, rel_coho_barcode_dim in enumerate(rel_coho_barcode):
            hom_barcode_dim = []
            for pair in rel_coho_barcode_dim:
                if pair[1] == np.inf:
                    hom_barcode_dim.append((N - 1 - pair[0], np.inf))
                else:
                    hom_barcode[dim - 1].append((N - 1 - pair[1],
                                                 N - 1 - pair[0]))
            hom_barcode.append(hom_barcode_dim)

    else:
        for dim, rel_coho_barcode_dim in enumerate(rel_coho_barcode):
            hom_barcode_dim = []
            for pair in rel_coho_barcode_dim:
                if pair[1] == np.inf:
                    hom_barcode_dim.append(
                        (filtration_values[N - 1 - pair[0]], np.inf)
                        )
                else:
                    hom_barcode[dim - 1].append(
                        (filtration_values[N - 1 - pair[1]],
                         filtration_values[N - 1 - pair[0]])
                        )
            hom_barcode.append(hom_barcode_dim)

    return hom_barcode


def to_values_barcode(rel_coho_barcode, N, filtration_values):
    values_barcode = []
    for dim, rel_coho_barcode_dim in enumerate(rel_coho_barcode):
        values_barcode_dim = []
        for pair in rel_coho_barcode_dim:
            if pair[1] == np.inf:
                values_barcode_dim.append(
                    (filtration_values[N - 1 - pair[0]], np.inf)
                )
            else:
                values_barcode[dim - 1].append(
                    (filtration_values[N - 1 - pair[0]],
                     filtration_values[N - 1 - pair[1]])
                )
        values_barcode.append(values_barcode_dim)

    return values_barcode


def check_agreement_with_gudhi(gudhi_barcode, barcode):
    max_dimension_gudhi = max([pers_info[0] for pers_info in gudhi_barcode])
    assert max_dimension_gudhi <= len(barcode) - 1

    for dim, barcode_dim in enumerate(barcode):
        gudhi_barcode_dim = sorted([
            pers_info[1] for pers_info in gudhi_barcode if pers_info[0] == dim
            ])
        assert gudhi_barcode_dim == sorted(barcode_dim), \
            f"Disagreement in degree {dim}"


@njit
def _symm_diff(x, y):
    n = len(x)
    m = len(y)
    result = []
    i = 0
    j = 0
    while (i < n) and (j < m):
        if x[i] < y[j]:
            result.append(x[i])
            i += 1
        elif y[j] < x[i]:
            result.append(y[j])
            j += 1
        else:
            i += 1
            j += 1

    while i < n:
        result.append(x[i])
        i += 1

    while j < m:
        result.append(y[j])
        j += 1

    return result


@njit
def _drop_elements(tup: tuple):
    for x in range(len(tup)):
        empty = tup[:-1]  # Not empty, but the right size and will be mutated
        idx = 0
        for i in range(len(tup)):
            if i != x:
                empty = tuple_setitem(empty, idx, tup[i])
                idx += 1
        yield empty
