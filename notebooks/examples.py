from itertools import combinations, chain
from random import shuffle
from gudhi import SimplexTree
from data import top_rp2, top_rp3, top_rp4, top_cp2


def make_simplices(top_faces):
    """It returns the simplices in a complex given its top dimensional
    simplices only.

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
    """It returns an ordering of the simplices of a simplicial complex defining a
    filtration of it.

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


def cone(simplices):
    """Returns a filtration of the cone of a given simplicial complex.

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
    """Returns a filtration of the suspension of a given simplicial complex.

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
    """Returns a filtration of the n-sphere.

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
    """Returns a filtration wedge of two simplicial complexes.

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


rp2 = make_filtration(make_simplices(top_rp2))
rp3 = make_filtration(make_simplices(top_rp3))
rp4 = make_filtration(make_simplices(top_rp4))
cp2 = make_filtration(make_simplices(top_cp2))
