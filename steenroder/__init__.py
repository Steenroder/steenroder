import numpy as np
from itertools import combinations
from numba import njit
from numba.typed import List


@njit
def _pivot(column):
    for i in range(len(column) - 1, -1, -1):
        if column[i]:
            return i
    return -1


@njit
def _are_pivots_same(column_1, column_2):
    # Assumes column_1 and column_2 have the same length
    for i in range(len(column_1) - 1, -1, -1):
        if column_1[i] or column_2[i]:
            if column_1[i] and column_2[i]:
                return True
            return False
    return False


def get_boundary(filtration):
    spx_filtration_idx = {tuple(v): idx for idx, v in enumerate(filtration)}
    boundary = np.zeros((len(filtration), len(filtration)), dtype=bool)
    for idx, spx in enumerate(filtration):
        faces_idxs = []
        try:
            faces_idxs = [spx_filtration_idx[spx[:j] + spx[j + 1:]]
                          for j in range(len(spx))]
        except KeyError:
            pass
        boundary[faces_idxs, idx] = True

    return boundary


def get_coboundary(filtration):
    coboundary = np.flip(get_boundary(filtration), axis=[0, 1]).transpose()
    return coboundary


@njit
def get_reduced_triangular(matrix, homology=False):
    """R = MV"""

    # # if a filtration is passed
    # if isinstance(matrix, tuple):
    #     matrix = get_boundary(matrix)
    #     if not homology:
    #         matrix = np.flip(matrix, axis=[0, 1]).transpose()

    # reduction steps
    n = matrix.shape[1]
    reduced = matrix.copy()
    triangular = np.zeros((n, n), dtype=np.bool_).T
    np.fill_diagonal(triangular, True)
    for j in range(n):
        i = j
        while i > 0:
            i -= 1
            if not np.any(reduced[:, j]):
                break
            else:
                if _are_pivots_same(reduced[:, j], reduced[:, i]):
                    reduced[:, j] = np.logical_xor(
                        reduced[:, i], reduced[:, j])
                    triangular[:, j] = np.logical_xor(
                        triangular[:, i], triangular[:, j])
                    i = j

    return reduced, triangular


def gen_coboundary_by_dim(filtration, maxdim=None):
    """Generates sparse coboundary matrices in order of increasing homology
    dimension"""
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

        yield (coboundary_keys_sorted,
               List([np.asarray(coboundary[x], dtype=np.int64)
                     for x in coboundary_keys_sorted]))
    
    yield None

    maxdim_splx = np.asarray(sorted(filtration_by_dim[maxdim - 1].keys()))[::-1]
    yield maxdim_splx, ([list()] * len(maxdim_splx), [[i] for i in maxdim_splx])


def get_reduced_triangular_sparse(matrices_by_dim):
    """R = MV"""
    ret = []
    for mat in matrices_by_dim:
        if mat is not None:
            ret.append((mat[0],
                        _get_reduced_triangular_sparse(mat[0], mat[1])))
        else:
            break
            
    ret.append(next(matrices_by_dim))

    return ret


@njit
def _get_reduced_triangular_sparse(idxs, matrix):
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
                reduced[j] = symm_diff(reduced[j], reduced[i])
                triangular[j] = symm_diff(triangular[j], triangular[i])
                i = j

    return reduced, triangular


def get_barcode(filtration, reduced=None):
    if reduced is None:
        coboundary = get_coboundary(filtration)
        reduced, _ = get_reduced_triangular(coboundary)
    pairs = []
    all_indices = []
    for j in range(len(filtration)):
        if np.any(reduced[:, j]):
            i = _pivot(reduced[:, j])
            pairs.append((i, j))
            all_indices += [i, j]

    for i in [j for j in range(len(filtration)) if j not in all_indices]:
        if not np.any(reduced[:, i]):
            pairs.append((i, np.inf))

    return sorted(pairs)


def get_barcode_from_sparse(filtration, idxs_reduced_triangular):
    N = len(filtration)
    pairs = []
    all_indices = set()
    for idxs, (reduced, triangular) in idxs_reduced_triangular:
        pairs_dim = []
        for i in range(len(idxs)):
            if reduced[i]:
                b = N - 1 - min(reduced[i])
                d = N - 1 - idxs[i]
                pairs_dim.append((b, d))
                all_indices |= {b, d}

        for i in range(len(idxs)):
            if N - 1 - idxs[i] not in all_indices:
                if not reduced[i]:
                    pairs_dim.append((N - 1 - idxs[i], np.inf))
            
        pairs.append(sorted(pairs_dim))

    return pairs


def filter_barcode_by_dim(barcode, filtration):
    max_dim = max([len(spx) for spx in filtration]) - 1
    barcode_by_dim = {i: [] for i in range(max_dim + 1)}
    for pair in barcode:
        d = len(filtration[-pair[0] - 1]) - 1
        barcode_by_dim[d] += [pair]
    return barcode_by_dim


def get_coho_reps(filtration, barcode=None, reduced=None, triangular=None):
    if reduced is None or triangular is None:
        coboundary = get_coboundary(filtration)
        reduced, triangular = get_reduced_triangular(coboundary)

    if barcode is None:
        barcode = get_barcode(filtration, reduced)

    coho_reps = np.empty((len(filtration), len(barcode)), dtype=bool)
    for col, pair in enumerate(barcode):
        if pair[1] < np.inf:
            coho_reps[:, col] = reduced[:, pair[1]]
        else:
            coho_reps[:, col] = triangular[:, pair[0]]
    return coho_reps


def get_coho_reps_from_sparse(filtration, barcode, idxs_reduced_triangular):
    N = len(filtration)
    coho_reps = []
    for dim, barcode_in_dim in enumerate(barcode):
        idxs, (reduced, triangular) = idxs_reduced_triangular[dim]
        coho_reps_in_dim = []
        for pair in barcode_in_dim:
            if pair[1] < np.inf:
                idx = (i for i, x in enumerate(idxs) if x == N - 1 - pair[1])
                coho_reps_in_dim.append([N - 1 - x for x in reduced[next(idx)]])
            else:
                idx = (i for i, x in enumerate(idxs) if x == N - 1 - pair[0])
                coho_reps_in_dim.append([N - 1 - x for x in triangular[next(idx)]])
                
        coho_reps.append(coho_reps_in_dim)

    return coho_reps


def vector_to_cochain(filtration, vector):
    cocycle = {filtration[len(filtration) - i - 1]
               for i in vector.nonzero()[0]}
    return cocycle


def cochain_to_vector(filtration, cochain):
    """returns a column vector shape (n,1)"""
    def simplex_to_index(spx):
        return len(filtration) - filtration.index(spx) - 1
    nonzero_indices = [simplex_to_index(spx) for spx in cochain]
    vector = np.zeros(shape=(len(filtration), 1), dtype=bool)
    vector[nonzero_indices] = True
    return vector


def STSQ(k, cocycle, filtration):
    """..."""
    answer = set()
    for pair in combinations(cocycle, 2):
        a, b = set(pair[0]), set(pair[1])
        u = sorted(a.union(b))
        if len(u) == len(a) + k and tuple(u) in filtration:
            a_bar, b_bar = a.difference(b), b.difference(a)
            u_bar = sorted(a_bar.union(b_bar))
            index = dict()
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


def get_st_reps(filtration, k, barcode=None, coho_reps=None):
    if barcode is None:
        barcode = get_barcode(filtration)
    if coho_reps is None:
        coho_reps = get_coho_reps(filtration, barcode)

    filtration_ = set(filtration)
    st_reps = np.zeros(coho_reps.shape, dtype=bool)
    for idx, rep in enumerate(np.transpose(coho_reps)):
        # from vector to cochain
        cocycle = vector_to_cochain(filtration, rep)
        cochain = STSQ(k, cocycle, filtration_)
        # cochain to vector
        st_reps[:, idx:idx + 1] = cochain_to_vector(filtration, cochain)

    return st_reps


def get_steenrod_matrix(k, coho_reps, barcode, filtration):
    filtration_ = set(filtration)
    dim = coho_reps.shape[0]
    steenrod_matrix = np.zeros((dim, dim), dtype=bool)
    for idx, rep in enumerate(np.transpose(coho_reps)):
        pos = barcode[idx][0]
        # from vector to cochain
        cocycle = vector_to_cochain(filtration, rep)
        cochain = STSQ(k, cocycle, filtration_)
        # cochain to vector
        steenrod_matrix[:, pos:pos + 1] = cochain_to_vector(filtration,
                                                            cochain)
    return steenrod_matrix


def get_pivots(matrix):
    n = matrix.shape[1]
    pivots = []
    for i in range(n):
        pivots.append(_pivot(matrix[:, i]))
    return pivots


def get_rank(matrix):
    sums = np.sum(matrix, axis=0)
    rank = len(sums.nonzero()[0])
    return rank


@njit
def reduce_vector(reduced, vector, num_col):
    i = -1
    while i >= -num_col:
        if not np.any(vector):
            break
        else:
            if _are_pivots_same(vector, reduced[:, i]):
                vector[:] = np.logical_xor(vector, reduced[:, i])
                i = 0
            i -= 1


@njit
def reduce_matrix(reduced, matrix):
    num_vector = matrix.shape[1]
    reducing = np.empty((reduced.shape[1] + num_vector, reduced.shape[0]),
                        dtype=reduced.dtype).T
    reducing[:, :reduced.shape[1]] = reduced

    for i in range(num_vector):
        reduce_vector(reducing, matrix[:, i], reduced.shape[1] + i)
        reducing[:, reduced.shape[1] + i] = matrix[:, i]


@njit
def get_steenrod_barcode(reduced, steenrod_matrix):
    dim = reduced.shape[0]
    alive = np.full(dim, True)

    R = reduced
    Q = steenrod_matrix
    barcode = []
    for j in range(dim):
        reduce_matrix(R[:, :j + 1], Q[:, :j + 1])
        for i in range(j + 1):
            if alive[i] and not np.any(Q[:, i]):
                alive[i] = False
                if j > i:
                    barcode.append((i, j))
    for i in range(len(alive)):
        if alive[i]:
            barcode.append((i, np.inf))

    return sorted([pair for pair in barcode if pair[1] > pair[0]])


def barcodes(k, filtration):
    """Serves as the main function"""

    coboundary = get_coboundary(filtration)
    reduced, triangular = get_reduced_triangular(coboundary)

    barcode = get_barcode(filtration, reduced=reduced)
    coho_reps = get_coho_reps(filtration, barcode=barcode,
                              reduced=reduced, triangular=triangular)
    steenrod_matrix = get_steenrod_matrix(k, coho_reps, barcode, filtration)
    st_barcode = get_steenrod_barcode(reduced, steenrod_matrix)

    return barcode, st_barcode


@njit
def symm_diff(arr1, arr2):
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
