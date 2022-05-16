import numpy as np
from itertools import combinations, chain
from random import shuffle
from gudhi import SimplexTree


def make_simplices(top_faces):
    """Return all simplices of a complex given its top ones.

    Parameters
    ----------
    top_faces : list of tuple of int
        Represents the top faces of a simplicial complex.

    Returns
    -------
    simplices : list of tuple of int
        Represent all simplices in the minimal complex with the given top
        dimensional simplices.

    """
    dim = len(top_faces[0]) - 1
    simplices = set()
    for top_face in top_faces:
        for d in range(dim + 1):
            simplices |= set(tuple(sorted(s)) for s in combinations(top_face, d + 1))
    return list(simplices)


def make_filtration(simplices):
    """Return a filtration of a given complex.

    Parameters
    ----------
    simplices : list of tuple of int
        Represents the simplices of a simplicial complex.

    Returns
    -------
    filtration : list of tuple of int
        Represents a filtration for the given simplicial complex.

    """
    st = SimplexTree()
    shuffle(simplices)
    for spx in simplices:
        st.insert(spx)
    st.make_filtration_non_decreasing()
    filtration = [tuple(pair[0]) for pair in st.get_filtration()]
    return filtration


def sort_filtration_by_dim(filtration, maxdim=None):
    """Organize an input filtration by dimension.

    Parameters
    ----------
    filtration : sequence of list-like of int
        Represents a simplex-wise filtration. Entry ``i`` is a list/tuple/set
        containing the integer indices of the vertices defining the ``i`` th
        simplex in the filtration.

    maxdim : int or None, optional, default: None
        Maximum simplex dimension to be included. ``None`` means that all
        simplices are included.

    Returns
    -------
    filtration_by_dim : list of list of ndarray
        For each dimension ``d``, a list of 2 aligned int arrays: the first is
        a 1D array containing the (ordered) positional indices of all
        ``d``-dimensional simplices in `filtration`; the second is a 2D array
        whose ``i``-th row is the (sorted) collection of vertices defining the
        ``i``-th ``d``-dimensional simplex.

    """
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


def cone(simplices):
    """Return a filtration of the cone of a complex.

    Parameters
    ----------
    simplices : list of tuple of int
        Represents the simplices of a simplicial complex.

    Returns
    -------
    cone : list of tuple of int
        Represents a filtration of the cone of the given simplicial complex.

    """
    vertices = set(v for v in chain.from_iterable(simplices))
    v = max(vertices) + 1
    new_simplices = simplices + [tuple(list(s) + [v]) for s in simplices] + [(v,)]
    cone = make_filtration(new_simplices)
    return cone


def suspension(simplices):
    """Return a filtration of the suspension of a complex.

    Parameters
    ----------
    simplices : list of tuple of int
        Represents the simplices of a simplicial complex.

    Returns
    -------
    suspension : list of tuple of int
        Represents a filtration of the suspension of the given simplicial complex.

    """
    vertices = set(v for v in chain.from_iterable(simplices))
    v, w = max(vertices) + 1, max(vertices) + 2
    new_simplices = list(simplices)
    new_simplices += [tuple(list(s) + [v]) for s in simplices] + [(v,)] +\
                     [tuple(list(s) + [w]) for s in simplices] + [(w,)]
    suspension = make_filtration(new_simplices)
    return suspension


def sphere(n):
    """Return a filtration of the n-sphere.

    Parameters
    ----------
    n : int
        The dimension of the sphere.

    Returns
    -------
    sphere : list of tuple of int
        Represents a filtration of the n-sphere.

    """
    top_faces = [s for s in combinations(range(1, n + 3), n + 1)]
    simplices = make_simplices(top_faces)
    return make_filtration(simplices)


def wedge(simplices_1, simplices_2):
    """Return a filtration of the wedge of two complexes.

    Parameters
    ----------
    simplices_1 : list of tuple of int
        Represents the first simplicial complex.

    simplices_2 : list of tuple of int
        Represents the second simplicial complex.

    Returns
    -------
    wedge : list of tuple of int
        Represents a filtration of the wedge of the given simplicial complexes.

    """
    vertices_1 = set(v for v in chain.from_iterable(simplices_1))
    v = max(vertices_1)
    simplices = list(simplices_1)
    simplices.remove((v,))
    for s in simplices_2:
        simplices.append(tuple(map(lambda i: i + v - 1, s)))
    wedge = make_filtration(simplices)
    return wedge
