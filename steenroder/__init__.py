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
def _standard_reduction(coboundary_matrix,
                        triangular,
                        pivots_lookup,
                        filtration_dim_plus_one_idxs):
    """R = MV"""
    n = len(coboundary_matrix)

    coboundary_matrix = coboundary_matrix[::-1]
    triangular = triangular[::-1]
    for j in range(n):
        highest_one = coboundary_matrix[j][0] if coboundary_matrix[j] else -1
        pivot_col = pivots_lookup[highest_one]
        while (highest_one != -1) and (pivot_col != -1):
            coboundary_matrix[j] = _symm_diff(
                coboundary_matrix[j][1:],
                coboundary_matrix[pivot_col][1:]
                )
            triangular[j] = _symm_diff(
                triangular[j],
                triangular[pivot_col]
                )
            highest_one = coboundary_matrix[j][0] if coboundary_matrix[j] else -1
            pivot_col = pivots_lookup[highest_one]
        if highest_one != -1:
            pivots_lookup[highest_one] = j
    
    for j in range(n):
        coboundary_matrix[j] = [filtration_dim_plus_one_idxs[k]
                                for k in coboundary_matrix[j]]

    return coboundary_matrix, triangular


@lru_cache
def get_reduced_triangular_single_dim(dim):
    len_tups_dim = dim + 1
    tuple_typ_dim = types.UniTuple(types.int64, len_tups_dim)
    len_tups_dim_plus_one = dim + 2

    @njit
    def inner(filtration_dim_idxs,
              filtration_dim_tups_array,
              filtration_dim_plus_one_idxs=None,
              filtration_dim_plus_one_tups_array=None):
        """R = MV"""
        spx_filtration_idx_dim = Dict.empty(tuple_typ_dim, types.int64)
        # Initialize reduced matrix as the coboundary matrix
        # TODO avoid this silly initializing and popping, needed now for type inference
        reduced = []
        triangular = []
        for i in range(len(filtration_dim_idxs)):
            spx = to_fixed_tuple(filtration_dim_tups_array[i], len_tups_dim)
            spx_filtration_idx_dim[spx] = i
            reduced.append([-1])
            reduced[-1].pop()
            triangular.append([filtration_dim_idxs[i]])

        if filtration_dim_plus_one_idxs is not None:
            for j in range(len(filtration_dim_plus_one_idxs)):
                spx = to_fixed_tuple(filtration_dim_plus_one_tups_array[j], len_tups_dim_plus_one)
                for face in _drop_elements(spx):
                    reduced[spx_filtration_idx_dim[face]].append(j)

            pivots_lookup = [-1] * len(filtration_dim_plus_one_idxs)

            reduced, triangular = _standard_reduction(reduced,
                                                      triangular,
                                                      pivots_lookup,
                                                      filtration_dim_plus_one_idxs)

        return spx_filtration_idx_dim, reduced, triangular

    return inner


def get_reduced_triangular(filtration_by_dim):
    maxdim = len(filtration_by_dim) - 1
    spxdict_idxs_reduced_triangular = []
    for dim in range(maxdim):
        reduction_in_dim = get_reduced_triangular_single_dim(dim)
        filtration_dim_idxs, filtration_dim_tups_array = filtration_by_dim[dim]
        filtration_dim_plus_one_idxs, filtration_dim_plus_one_tups_array = filtration_by_dim[dim + 1]
        spx_filtration_idx_dim, reduced, triangular = reduction_in_dim(
            filtration_dim_idxs,
            filtration_dim_tups_array,
            filtration_dim_plus_one_idxs,
            filtration_dim_plus_one_tups_array
            )
        spxdict_idxs_reduced_triangular.append((spx_filtration_idx_dim,
                                                filtration_dim_idxs[::-1],
                                                reduced,
                                                triangular))

    reduction_in_dim = get_reduced_triangular_single_dim(maxdim)
    filtration_dim_idxs, filtration_dim_tups_array = filtration_by_dim[maxdim]
    spx_filtration_idx_dim, reduced, triangular = reduction_in_dim(
        filtration_dim_idxs,
        filtration_dim_tups_array,
        None,
        None
        )
    spxdict_idxs_reduced_triangular.append((spx_filtration_idx_dim,
                                            filtration_dim_idxs[::-1],
                                            reduced,
                                            triangular))

    return spxdict_idxs_reduced_triangular


def get_barcode(filtration, spxdict_idxs_reduced_triangular):
    N = len(filtration)
    pairs = []
    all_indices = set()
    
    _, idxs, reduced, _ = spxdict_idxs_reduced_triangular[0]
    pairs_dim = []
    for i in range(len(idxs)):
        if N - 1 - idxs[i] not in all_indices:
            if not reduced[i]:
                pairs_dim.append((N - 1 - idxs[i], np.inf))
    pairs.append(sorted(pairs_dim))
    
    for dim in range(1, len(spxdict_idxs_reduced_triangular)):
        _, idxs, reduced, _ = spxdict_idxs_reduced_triangular[dim]
        _, idxs_prev, reduced_prev, _ = spxdict_idxs_reduced_triangular[dim - 1]

        pairs_dim = []
        for i in range(len(idxs_prev)):
            if reduced_prev[i]:
                b = N - 1 - reduced_prev[i][0]
                d = N - 1 - idxs_prev[i]
                pairs_dim.append((b, d))
                all_indices |= {b, d}

        for i in range(len(idxs)):
            if N - 1 - idxs[i] not in all_indices:
                if not reduced[i]:
                    pairs_dim.append((N - 1 - idxs[i], np.inf))
            
        pairs.append(sorted(pairs_dim))

    return pairs


def get_coho_reps(filtration, barcode, spxdict_idxs_reduced_triangular):
    N = len(filtration)
    coho_reps = []
    
    _, idxs, reduced, triangular = spxdict_idxs_reduced_triangular[0]
    coho_reps_in_dim = []
    for pair in barcode[0]:
        idx = np.flatnonzero(idxs == N - 1 - pair[0])[0]
        coho_reps_in_dim.append([N - 1 - x for x in triangular[idx]])
    coho_reps.append(coho_reps_in_dim)
    
    for dim in range(1, len(barcode)):
        barcode_in_dim = barcode[dim]
        _, idxs, reduced, triangular = spxdict_idxs_reduced_triangular[dim]
        _, idxs_prev, reduced_prev, _ = spxdict_idxs_reduced_triangular[dim - 1]

        coho_reps_in_dim = []
        for pair in barcode_in_dim:
            if pair[1] < np.inf:
                idx = np.flatnonzero(idxs_prev == N - 1 - pair[1])[0]
                coho_reps_in_dim.append([N - 1 - x for x in reduced_prev[idx]])
            else:
                idx = np.flatnonzero(idxs == N - 1 - pair[0])[0]
                coho_reps_in_dim.append([N - 1 - x for x in triangular[idx]])
                
        coho_reps.append(coho_reps_in_dim)

    return coho_reps


def STSQ(k, cocycle, filtration):
    """..."""
    answer = set()
    for pair in combinations(cocycle, 2):
        a, b = set(pair[0]), set(pair[1])
        u = sorted(a.union(b))
        if len(u) == len(a) + k and tuple(u) in filtration:
            a_bar, b_bar = a.difference(b), b.difference(a)
            u_bar = sorted(a_bar.union(b_bar))
            index = {}
            for v in a_bar.union(b_bar):
                pos = u.index(v)
                pos_bar = u_bar.index(v)
                index[v] = (pos + pos_bar) % 2
            index_a = {index[v] for v in a_bar}
            index_b = {index[w] for w in b_bar}
            if (index_a == {0} and index_b == {1}
                    or index_a == {1} and index_b == {0}):
                answer ^= {tuple(u)}

    return answer


def get_steenrod_matrix(k, coho_reps, filtration, spxdict_idxs_reduced_triangular):
    N = len(filtration)
    steenrod_matrix = [list()] * k
    for dim, coho_reps_in_dim in enumerate(coho_reps[:-1]):
        steenrod_matrix.append([])
        for i, rep in enumerate(coho_reps_in_dim):
            cocycle = set(filtration[N - 1 - j] for j in rep)
            spx_filtration_idx_dim, idxs, _, _ = spxdict_idxs_reduced_triangular[dim + k]
            cochain = STSQ(k, cocycle, spx_filtration_idx_dim)
            steenrod_matrix[dim + k].append([N - 1 - idxs[-1 - spx_filtration_idx_dim[spx]]
                                             for spx in cochain])
        
    return steenrod_matrix


def get_steenrod_barcode(k, steenrod_matrix, spxdict_idxs_reduced_triangular, barcode, filtration):
    N = len(filtration)

    st_barcode = [list()] * k
    for dim in range(k, len(steenrod_matrix)):
        if steenrod_matrix[dim]:
            births_dim = np.asarray([N - 1 - pair[0] for pair in barcode[dim - k]], dtype=np.int64)
            _, idxs_prev, reduced_prev, _ = spxdict_idxs_reduced_triangular[dim - 1]
            reduced_prev = List([np.asarray(x, dtype=np.int64) for x in reduced_prev])
            steenrod_matrix_dim = List([np.sort([N - 1 - x for x in cochain]).astype(np.int64)
                                        for cochain in steenrod_matrix[dim]])
            st_barcode.append([pair if pair[1] != -1 else (pair[0], np.inf)
                               for pair in _get_steenrod_barcode_in_dim(steenrod_matrix_dim, idxs_prev, reduced_prev, births_dim, N)])
        else:
            st_barcode.append([])

    return st_barcode


@njit
def _get_steenrod_barcode_in_dim(steenrod_matrix_dim, idxs_prev, reduced_prev, births_dim, N):
    augmented = []
    for i in range(len(reduced_prev) - 1, -1, -1):
        augmented.append([x for x in reduced_prev[i]])

    for i in range(len(steenrod_matrix_dim)):
        augmented.append([x for x in steenrod_matrix_dim[i]])

    alive = [True] * len(births_dim)
    n = len(reduced_prev)
    st_barcode_dim = []

    j = 0
    for i, idx in enumerate(idxs_prev):
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
                        augmented[ii] = _symm_diff(augmented[iii][1:], augmented[ii][1:])
                        iii = ii

            if alive[ii - n] and (not augmented[ii]):
                alive[ii - n] = False
                if idx < births_dim[ii - n]:
                    st_barcode_dim.append((N - 1 - births_dim[ii - n], N - 1 - idx))

    for i in range(len(alive)):
        if alive[i]:
            st_barcode_dim.append((N - 1 - births_dim[i], -1))

    return st_barcode_dim
                        

def barcodes(k, filtration, maxdim=None):
    """Serves as the main function"""
    filtration_by_dim = sort_filtration_by_dim(filtration, maxdim=maxdim)
    spxdict_idxs_reduced_triangular = get_reduced_triangular(filtration_by_dim)
    barcode = get_barcode(filtration, spxdict_idxs_reduced_triangular)
    coho_reps = get_coho_reps(filtration, barcode, spxdict_idxs_reduced_triangular)
    steenrod_matrix = get_steenrod_matrix(k, coho_reps, filtration, spxdict_idxs_reduced_triangular)
    st_barcode = get_steenrod_barcode(k, steenrod_matrix, spxdict_idxs_reduced_triangular, barcode, filtration)

    return barcode, st_barcode


def remove_trivial_bars(barcode, filtration):
    """Note: filtration is as returned by GUDHI, i.e. [(simplex_tuple, filtration value), ...]"""
    N = len(filtration)
    barcode_vals = []
    for barcode_in_dim in barcode:
        barcode_vals_in_dim = []
        for tup in barcode_in_dim:
            if not np.isinf(tup[1]):
                candidate = (filtration[N - 1 - tup[0]][1], filtration[N - 1 - tup[1]][1])
                if candidate[0] != candidate[1]:
                    barcode_vals_in_dim.append(candidate)
            else:
                barcode_vals_in_dim.append((filtration[N - 1 - tup[0]][1], -np.inf))
        barcode_vals.append(barcode_vals_in_dim)
    
    return barcode_vals


@njit
def _symm_diff(arr1, arr2):
    n = len(arr1)
    m = len(arr2)
    result = []
    i = 0
    j = 0
    while (i < n) and (j < m):
        if arr1[i] < arr2[j]:
            result.append(arr1[i])
            i += 1
        elif arr2[j] < arr1[i]:
            result.append(arr2[j])
            j += 1
        else:
            i += 1
            j += 1

    while i < n:
        result.append(arr1[i])
        i += 1

    while j < m:
        result.append(arr2[j])
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
