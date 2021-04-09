from itertools import combinations

import numpy as np
from numba import njit
from numba.typed import List


def sort_filtration_by_dim(filtration, maxdim=None):
    if maxdim is None:
        maxdim = max(map(len, filtration))
    filtration_by_dim = [{} for i in range(maxdim + 1)]
    spx_filtration_idx_by_dim = [{} for i in range(maxdim + 1)]
    for idx, v in enumerate(filtration):
        v_t = tuple(sorted(v))
        dim = len(v_t) - 1
        if dim <= maxdim:
            filtration_by_dim[dim][idx] = v_t
            spx_filtration_idx_by_dim[dim][v_t] = idx
    
    return filtration_by_dim, spx_filtration_idx_by_dim


def get_reduced_triangular(filtration_by_dim, spx_filtration_idx_by_dim):
    """R = MV"""
    maxdim = len(filtration_by_dim) - 1
    idxs_reduced_triangular = []
    
    for dim in range(maxdim - 1):
        filtration_dim_plus_one = filtration_by_dim[dim + 1]
        spx_filtration_idx_dim = spx_filtration_idx_by_dim[dim]
        coboundary = {}
        for idx, spx in filtration_dim_plus_one.items():
            for j in range(dim + 2):
                face_idx = spx_filtration_idx_dim[spx[:j] + spx[j + 1:]]
                if face_idx not in coboundary:
                    coboundary[face_idx] = [idx]
                else:
                    coboundary[face_idx].append(idx)

        coboundary_keys_sorted = np.asarray(sorted(coboundary.keys()))[::-1]
        coboundary_vals_sorted = List([np.asarray(coboundary[x], dtype=np.int64)
                                       for x in coboundary_keys_sorted])
        
        idxs_reduced_triangular.append(
            (coboundary_keys_sorted,
             _get_reduced_triangular(coboundary_keys_sorted, coboundary_vals_sorted))
            )

    # Special treatment for top dimension
    maxdim_splx = np.asarray(sorted(filtration_by_dim[maxdim - 1].keys()))[::-1]
    idxs_reduced_triangular.append(
        (maxdim_splx,
         ([list()] * len(maxdim_splx), [[i] for i in maxdim_splx]))
        )

    return idxs_reduced_triangular


@njit
def _get_reduced_triangular(idxs, matrix):
    """R = MV"""

    n = len(idxs)
    reduced = []
    triangular = []
    for j in range(n):
        reduced.append([x for x in matrix[j]])
        triangular.append([idxs[j]])
        i = j
        while i:
            i -= 1
            if not reduced[j]:
                break
            elif not reduced[i]:
                continue
            elif reduced[j][0] == reduced[i][0]:
                reduced[j] = _symm_diff(reduced[j][1:], reduced[i][1:])
                triangular[j] = _symm_diff(triangular[j], triangular[i])
                i = j

    return reduced, triangular


def get_barcode(filtration, idxs_reduced_triangular):
    N = len(filtration)
    pairs = []
    all_indices = set()
    
    idxs, (reduced, triangular) = idxs_reduced_triangular[0]
    pairs_dim = []
    for i in range(len(idxs)):
        if N - 1 - idxs[i] not in all_indices:
            if not reduced[i]:
                pairs_dim.append((N - 1 - idxs[i], np.inf))
    pairs.append(sorted(pairs_dim))
    
    for dim in range(1, len(idxs_reduced_triangular)):
        idxs, (reduced, _) = idxs_reduced_triangular[dim]
        idxs_prev, (reduced_prev, _) = idxs_reduced_triangular[dim - 1]

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


def get_coho_reps(filtration, barcode, idxs_reduced_triangular):
    N = len(filtration)
    coho_reps = []
    
    idxs, (reduced, triangular) = idxs_reduced_triangular[0]
    coho_reps_in_dim = []
    for pair in barcode[0]:
        idx = np.flatnonzero(idxs == N - 1 - pair[0])[0]
        coho_reps_in_dim.append([N - 1 - x for x in triangular[idx]])
    coho_reps.append(coho_reps_in_dim)
    
    for dim in range(1, len(barcode)):
        barcode_in_dim = barcode[dim]
        idxs, (reduced, triangular) = idxs_reduced_triangular[dim]
        idxs_prev, (reduced_prev, _) = idxs_reduced_triangular[dim - 1]

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


def get_steenrod_matrix(k, coho_reps, filtration, spx_filtration_idx_by_dim):
    N = len(filtration)
    filtration_ = set(filtration)
    steenrod_matrix = [list()] * k
    for dim, coho_reps_in_dim in enumerate(coho_reps[:-1]):
        steenrod_matrix.append([])
        for i, rep in enumerate(coho_reps_in_dim):
            cocycle = set(filtration[N - 1 - j] for j in rep)
            cochain = STSQ(k, cocycle, filtration_)
            steenrod_matrix[dim + k].append([N - 1 - spx_filtration_idx_by_dim[dim + k][spx] for spx in cochain])
        
    return steenrod_matrix


def get_steenrod_barcode(k, steenrod_matrix, idxs_reduced_triangular, barcode, filtration):
    N = len(filtration)

    st_barcode = [list()] * k
    for dim in range(k, len(steenrod_matrix)):
        if steenrod_matrix[dim]:
            births_dim = np.asarray([N - 1 - pair[0] for pair in barcode[dim - k]], dtype=np.int64)
            idxs_prev, (reduced_prev, _) = idxs_reduced_triangular[dim - 1]
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
    filtration_by_dim, spx_filtration_idx_by_dim = sort_filtration_by_dim(filtration, maxdim=maxdim)
    idxs_reduced_triangular = get_reduced_triangular(filtration_by_dim, spx_filtration_idx_by_dim)
    barcode = get_barcode(filtration, idxs_reduced_triangular)
    coho_reps = get_coho_reps(filtration, barcode, idxs_reduced_triangular)
    steenrod_matrix = get_steenrod_matrix(k, coho_reps, filtration, spx_filtration_idx_by_dim)
    st_barcode = get_steenrod_barcode(k, steenrod_matrix, idxs_reduced_triangular, barcode, filtration)

    return barcode, st_barcode


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
