import time
from collections.abc import Sequence
from functools import lru_cache

import numba as nb
import numpy as np
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.unsafe.ndarray import to_fixed_tuple

list_of_int64_typ = nb.types.List(nb.int64)
int64_2d_array_typ = nb.types.Array(nb.int64, 2, "C")


def sort_filtration_by_dim(filtration: Sequence[tuple[int]],
                           maxdim=None) -> list[list[np.ndarray, np.ndarray]]:
    if maxdim is None:
        maxdim = max(map(len, filtration)) - 1

    filtration_by_dim = [[] for _ in range(maxdim + 1)]
    for i, spx in enumerate(filtration):
        spx_tup = tuple(sorted(spx))
        dim = len(spx_tup) - 1
        if dim <= maxdim:
            filtration_by_dim[dim].append([i, spx_tup])

    for dim, filtr in enumerate(filtration_by_dim):
        filtration_by_dim[dim] = [np.asarray(x, dtype=np.int64)
                                  for x in zip(*filtr)]

    return filtration_by_dim


@nb.njit
def _twist_reduction(coboundary, triangular, pivots_lookup):
    """R = MV"""
    n = len(coboundary)

    rel_idxs_to_clear =[]
    for j in range(n - 1, -1, -1):
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
            rel_idxs_to_clear.append(highest_one)

    return np.asarray(rel_idxs_to_clear, dtype=np.int64)


@lru_cache
def _reduce_single_dim(dim):
    len_tups_dim = dim + 1
    tuple_typ_dim = nb.types.UniTuple(nb.int64, len_tups_dim)
    len_tups_next_dim = dim + 2

    @nb.njit
    def _inner_reduce_single_dim(idxs_dim, tups_dim, rel_idxs_to_clear,
                                 idxs_next_dim=None, tups_next_dim=None):
        """R = MV"""
        # 1) Construct sp2idx_dim as a dict simplex: relative (i.e.
        # in-dimension) index
        # 2) Initialize type of reduced_dim (needed for type inference)
        # 3) Construct triangular_dim, with entries denoting relative (i.e.
        # in-dimension) indices
        spx2idx_dim = nb.typed.Dict.empty(tuple_typ_dim, nb.int64)
        reduced_dim = nb.typed.List.empty_list(list_of_int64_typ)
        triangular_dim = nb.typed.List.empty_list(list_of_int64_typ)
        for i in range(len(idxs_dim)):
            spx = to_fixed_tuple(tups_dim[i], len_tups_dim)
            spx2idx_dim[spx] = i
            reduced_dim.append([nb.int64(x) for x in range(0)])
            triangular_dim.append([i])

        # Populate reduced_dim as the coboundary matrix and apply clearing
        # WARNING: Column entries denote relative (i.e. in-dimension) indices!
        if idxs_next_dim is not None:
            for j in range(len(idxs_next_dim)):
                spx = to_fixed_tuple(tups_next_dim[j], len_tups_next_dim)
                for face in _drop_elements(spx):
                    reduced_dim[spx2idx_dim[face]].append(j)

            for rel_idx in rel_idxs_to_clear:
                reduced_dim[rel_idx] = [nb.int64(x) for x in range(0)]

            pivots_lookup = np.full(len(idxs_next_dim), -1, dtype=np.int64)

            rel_idxs_to_clear = _twist_reduction(reduced_dim, triangular_dim,
                                                 pivots_lookup)

        return spx2idx_dim, reduced_dim, triangular_dim, rel_idxs_to_clear

    return _inner_reduce_single_dim


def get_reduced_triangular(filtration_by_dim: list[list[np.ndarray]]) \
        -> tuple[tuple[nb.typed.Dict],
                 tuple[np.ndarray],
                 tuple[nb.typed.List[list]],
                 tuple[nb.typed.List[list]]]:
    maxdim = len(filtration_by_dim) - 1
    # Initialize relative (i.e. in-dimension) indices to clear, as an empty
    # int array in dim 0
    rel_idxs_to_clear = np.empty(0, dtype=np.int64)
    spx2idx_idxs_reduced_triangular = []
    for dim in range(maxdim):
        reduction_dim = _reduce_single_dim(dim)
        idxs_dim, tups_dim = filtration_by_dim[dim]
        idxs_next_dim, tups_next_dim = filtration_by_dim[dim + 1]
        spx2idx_dim, reduced_dim, triangular_dim, rel_idxs_to_clear = \
            reduction_dim(idxs_dim,
                          tups_dim,
                          rel_idxs_to_clear,
                          idxs_next_dim=idxs_next_dim,
                          tups_next_dim=tups_next_dim)
        spx2idx_idxs_reduced_triangular.append((spx2idx_dim,
                                                idxs_dim,
                                                reduced_dim,
                                                triangular_dim))

    reduction_dim = _reduce_single_dim(maxdim)
    idxs_dim, tups_dim = filtration_by_dim[maxdim]
    spx2idx_dim, reduced_dim, triangular_dim, _ = \
        reduction_dim(idxs_dim, tups_dim, rel_idxs_to_clear)
    spx2idx_idxs_reduced_triangular.append((spx2idx_dim,
                                            idxs_dim,
                                            reduced_dim,
                                            triangular_dim))

    return tuple(zip(*spx2idx_idxs_reduced_triangular))


@nb.njit
def get_barcode_and_coho_reps(
        idxs: tuple[np.ndarray],
        reduced: tuple[nb.typed.List[list]],
        triangular: tuple[nb.typed.List[list]],
        filtration_values=None
        ) -> tuple[list[np.ndarray], list[nb.typed.List[list[int]]]]:
    barcode = []
    coho_reps = []

    if filtration_values is None:
        pairs_0 = []
        coho_reps_0 = nb.typed.List.empty_list(list_of_int64_typ)
        for i in range(len(idxs[0])):
            if not reduced[0][i]:
                pairs_0.append([-1, idxs[0][i]])
                coho_reps_0.append(triangular[0][i])
        barcode.append(np.asarray(sorted(pairs_0)))
        coho_reps.append(coho_reps_0)

        for dim in range(1, len(idxs)):
            all_birth_indices = set()
            pairs_dim = []
            coho_reps_dim = nb.typed.List.empty_list(list_of_int64_typ)
            for i in range(len(idxs[dim - 1])):
                if reduced[dim - 1][i]:
                    b = idxs[dim][reduced[dim - 1][i][0]]
                    d = idxs[dim - 1][i]
                    pairs_dim.append([d, b])
                    coho_reps_dim.append(reduced[dim - 1][i])
                    all_birth_indices.add(b)

            for i in range(len(idxs[dim])):
                if idxs[dim][i] not in all_birth_indices:
                    if not reduced[dim][i]:
                        pairs_dim.append([-1, idxs[dim][i]])
                        coho_reps_dim.append(triangular[dim][i])

            barcode.append(np.asarray(sorted(pairs_dim)))
            coho_reps.append(coho_reps_dim)

    else:
        pairs_0 = []
        coho_reps_0 = nb.typed.List.empty_list(list_of_int64_typ)
        for i in range(len(idxs[0])):
            if not reduced[0][i]:
                pairs_0.append([-1, idxs[0][i]])
        barcode.append(np.asarray(sorted(pairs_0)))
        coho_reps.append(coho_reps_0)

        for dim in range(1, len(idxs)):
            all_birth_indices = set()
            pairs_dim = []
            coho_reps_dim = nb.typed.List.empty_list(list_of_int64_typ)
            for i in range(len(idxs[dim - 1])):
                if reduced[dim - 1][i]:
                    b = idxs[dim][reduced[dim - 1][i][0]]
                    d = idxs[dim - 1][i]
                    if filtration_values[b] != filtration_values[d]:
                        pairs_dim.append([d, b])
                        coho_reps_dim.append(reduced[dim - 1][i])
                    all_birth_indices.add(b)

            for i in range(len(idxs[dim])):
                if idxs[dim][i] not in all_birth_indices:
                    if not reduced[dim][i]:
                        pairs_dim.append([-1, idxs[dim][i]])
                        coho_reps_dim.append(triangular[dim][i])

            barcode.append(np.asarray(sorted(pairs_dim)))
            coho_reps.append(coho_reps_dim)

    return barcode, coho_reps


@lru_cache
def _populate_steenrod_matrix_single_dim(dim_plus_k):
    length = dim_plus_k + 1

    @nb.njit
    def _inner(steenrod_matrix_dim_plus_k, coho_reps_dim, tups_dim,
               spx2idx_dim_plus_k):
        def STSQ(cocycle):
            """..."""
            answer = set(
                [to_fixed_tuple(np.empty(length, dtype=np.int64), length)]
                )
            answer.pop()
            for i in range(len(cocycle)):
                for j in range(i + 1, len(cocycle)):
                    a, b = set(cocycle[i]), set(cocycle[j])
                    u = a.union(b)
                    if len(u) == length:
                        u_tuple = to_fixed_tuple(np.asarray(sorted(u)), length)
                        if u_tuple in spx2idx_dim_plus_k:
                            a_bar, b_bar = a.difference(b), b.difference(a)
                            u_bar = sorted(a_bar.union(b_bar))
                            index = {}
                            for v in a_bar.union(b_bar):
                                pos = u_tuple.index(v)
                                pos_bar = u_bar.index(v)
                                index[v] = (pos + pos_bar) % 2
                            index_a = set()
                            index_b = set()
                            for v in a_bar:
                                index_a.add(index[v])
                            for w in b_bar:
                                index_b.add(index[w])
                            if (index_a == set([0])
                                and index_b == set([1])) \
                                    or (index_a == set([1])
                                        and index_b == set([0])):
                                answer ^= {u_tuple}

            return answer

        for i, rep in enumerate(coho_reps_dim):
            cocycle = tups_dim[np.asarray(rep)]
            cochain = STSQ(cocycle)
            steenrod_matrix_dim_plus_k.append(
                sorted([spx2idx_dim_plus_k[spx] for spx in cochain])
                )

    return _inner


def get_steenrod_matrix(k, coho_reps, filtration_by_dim, spx2idx):
    num_dimensions = len(filtration_by_dim)

    @nb.njit
    def _initialize_steenrod_matrix():
        return [nb.typed.List.empty_list(list_of_int64_typ)
                for _ in range(num_dimensions)]

    steenrod_matrix = _initialize_steenrod_matrix()

    for dim, coho_reps_dim in enumerate(coho_reps[:-k]):
        dim_plus_k = dim + k
        tups_dim = filtration_by_dim[dim][1]
        spx2idx_dim_plus_k = spx2idx[dim + k]
        steenrod_matrix_dim_plus_k = steenrod_matrix[dim_plus_k]
        populate_steenrod_matrix_single_dim = \
            _populate_steenrod_matrix_single_dim(dim_plus_k)
        populate_steenrod_matrix_single_dim(steenrod_matrix_dim_plus_k,
                                            coho_reps_dim, tups_dim,
                                            spx2idx_dim_plus_k)
        
    return steenrod_matrix


@nb.njit
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
    tic = time.time()
    N = len(filtration)
    filtration_by_dim = sort_filtration_by_dim(filtration, maxdim=maxdim)
    spx2idx_idxs_reduced_triangular = get_reduced_triangular(filtration_by_dim)
    barcode = get_barcode(N, spx2idx_idxs_reduced_triangular,
                          filtration_values=filtration_values)
    toc = time.time()
    print(f"Usual barcode computed, time taken: {toc - tic}")
    tic = time.time()
    coho_reps = get_coho_reps(N, barcode, spx2idx_idxs_reduced_triangular)
    steenrod_matrix = get_steenrod_matrix(k, coho_reps, filtration,
                                          spx2idx_idxs_reduced_triangular)
    toc = time.time()
    print(f"Steenrod matrix computed, time taken: {toc - tic}")
    tic = time.time()
    st_barcode = get_steenrod_barcode(k, steenrod_matrix,
                                      spx2idx_idxs_reduced_triangular, barcode,
                                      N, filtration_values=filtration_values)
    toc = time.time()
    print(f"Steenrod barcode computed, time taken: {toc - tic}")

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


@nb.njit
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


@nb.njit
def _drop_elements(tup: tuple):
    for x in range(len(tup)):
        empty = tup[:-1]  # Not empty, but the right size and will be mutated
        idx = 0
        for i in range(len(tup)):
            if i != x:
                empty = tuple_setitem(empty, idx, tup[i])
                idx += 1
        yield empty
