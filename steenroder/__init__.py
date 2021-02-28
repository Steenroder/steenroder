import numpy as np
from itertools import combinations
from numba import njit


@njit
def _pivot(column):
    nonzero = column.nonzero()[0]
    if len(nonzero):
        return max(nonzero)
    return -1


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
    triangular = np.zeros((n, n), dtype=np.bool_)
    np.fill_diagonal(triangular, True)
    for j in range(n):
        i = j
        while i > 0:
            i -= 1
            if not np.any(reduced[:, j]):
                break
            else:
                piv_j = _pivot(reduced[:, j])
                piv_i = _pivot(reduced[:, i])

                if piv_i == piv_j:
                    reduced[:, j] = np.logical_xor(
                        reduced[:, i], reduced[:, j])
                    triangular[:, j] = np.logical_xor(
                        triangular[:, i], triangular[:, j])
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
                answer.add(tuple(u))

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
def reduce_vector(reduced, vector):
    num_col = reduced.shape[1]
    i = -1
    while i >= -num_col:
        if not np.any(vector):
            break
        else:
            piv_v = _pivot(vector)
            piv_i = _pivot(reduced[:, i])

            if piv_i == piv_v:
                vector[:, 0] = np.logical_xor(reduced[:, i], vector[:, 0])
                i = 0
            i -= 1


@njit
def reduce_matrix(reduced, matrix):
    num_vector = matrix.shape[1]
    reducing = reduced.copy()

    for i in range(num_vector):
        reduce_vector(reducing, matrix[:, i:i + 1])
        reducing = np.concatenate((reducing, matrix[:, i:i + 1]), axis=1)


@njit
def get_steenrod_barcode(reduced, steenrod_matrix):
    dim = reduced.shape[0]
    alive = {}
    for i in range(dim):
        alive[i] = True

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
    for i in alive:
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
