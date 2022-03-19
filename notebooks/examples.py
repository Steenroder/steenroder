from itertools import combinations, chain
from random import shuffle
from gudhi import SimplexTree


def make_simplices(top_faces):
    '''Given the top faces of a simplicial complex it returns a list of all simplices in
    it'''
    dim = len(top_faces[0]) - 1
    simplices = set()
    for top_face in top_faces:
        for d in range(dim + 1):
            simplices |= set(tuple(sorted(s)) for s in combinations(top_face, d + 1))
    return list(simplices)


def make_filtration(simplices):
    '''Given a list of simplices in a simplicial complex it returns a list representing
    a simplicial complex, it returns a reordered of the list representing a filtration
    of it'''
    st = SimplexTree()
    shuffle(simplices)
    for spx in simplices:
        st.insert(spx)
    st.make_filtration_non_decreasing()
    return [tuple(pair[0]) for pair in st.get_filtration()]


def cone(simplices):
    '''Given a list of simplices representing a simplicial complex, it returns a
    list representing a filtration of its cone'''
    vertices = set(v for v in chain.from_iterable(simplices))
    v = max(vertices) + 1
    new_simplices = simplices + [tuple(list(s) + [v]) for s in simplices] + [(v,)]
    return make_filtration(new_simplices)


def suspension(simplices):
    '''Given a list of simplices representing a simplicial complex, it returns a
    list representing a filtration of its suspension'''
    vertices = set(v for v in chain.from_iterable(simplices))
    v, w = max(vertices) + 1, max(vertices) + 2
    simplices = list(simplices)
    simplices += [tuple(list(s) + [v]) for s in simplices] + [(v,)] +\
                 [tuple(list(s) + [w]) for s in simplices] + [(w,)]
    return make_filtration(simplices)


def sphere(n):
    '''Given an positive integer n, it returns a list representing a filtration of a
    simplicial complex model of the n-sphere'''
    top_faces = [s for s in combinations(range(1, n + 3), n + 1)]
    simplices = make_simplices(top_faces)
    return make_filtration(simplices)


def wedge(simplices_1, simplices_2):
    '''Given lists representing two simplicial complexes, it returns a list representing
    a filtration of a simplicial complex model of their wedge'''
    vertices_1 = set(v for v in chain.from_iterable(simplices_1))
    v = max(vertices_1)
    simplices = list(simplices_1)
    simplices.remove((v,))
    for s in simplices_2:
        simplices.append(tuple(map(lambda i: i + v - 1, s)))
    return make_filtration(simplices)


### RP2 ###

top_rp2 = (
    (1, 2, 4), (2, 3, 4), (1, 3, 5), (2, 3, 5), (1, 4, 5),
    (1, 2, 6), (1, 3, 6), (3, 4, 6), (2, 5, 6), (4, 5, 6)
)

### RP3 ###

top_rp3 = (
    (1, 2, 3, 4), (1, 2, 3, 5), (1, 3, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6),
    (1, 4, 5, 6), (2, 3, 4, 7), (2, 3, 5, 8), (2, 3, 7, 8), (2, 4, 6, 9),
    (2, 4, 7, 9), (2, 5, 6, 10), (2, 5, 8, 10), (2, 6, 9, 10), (2, 7, 8, 11),
    (2, 7, 9, 11), (2, 8, 10, 11), (2, 9, 10, 11), (3, 4, 5, 12), (3, 4, 7, 12),
    (3, 5, 8, 12), (9, 10, 11, 12), (3, 7, 8, 13), (6, 9, 10, 13), (3, 7, 12, 13),
    (3, 8, 12, 13), (8, 9, 12, 13), (7, 10, 12, 13), (9, 10, 12, 13), (4, 5, 6, 14),
    (4, 6, 9, 14), (5, 6, 10, 14), (7, 8, 11, 14), (4, 5, 12, 14), (7, 8, 13, 14),
    (6, 9, 13, 14), (8, 9, 13, 14), (6, 10, 13, 14), (7, 10, 13, 14), (4, 7, 9, 15),
    (8, 10, 11, 15), (4, 7, 12, 15), (7, 10, 12, 15), (10, 11, 12, 15), (4, 9, 14, 15),
    (8, 9, 14, 15), (8, 11, 14, 15), (4, 12, 14, 15), (11, 12, 14, 15), (5, 8, 10, 16),
    (7, 9, 11, 16), (5, 8, 12, 16), (8, 9, 12, 16), (9, 11, 12, 16), (5, 10, 14, 16),
    (7, 10, 14, 16), (7, 11, 14, 16), (5, 12, 14, 16), (11, 12, 14, 16), (7, 9, 15, 16),
    (8, 9, 15, 16), (7, 10, 15, 16), (8, 10, 15, 16)
)

### RP4 ###

top_rp4 = (
    (1, 2, 4, 5, 11), (3, 4, 6, 7, 11), (1, 3, 8, 9, 11), (2, 6, 8, 10, 11), (5, 7, 9, 10, 11),
    (4, 5, 7, 8, 12), (2, 4, 6, 9, 12), (1, 3, 8, 9, 12), (2, 3, 5, 10, 12), (1, 6, 7, 10, 12),
    (2, 4, 5, 11, 12), (2, 4, 6, 11, 12), (4, 5, 7, 11, 12), (4, 6, 7, 11, 12), (2, 5, 10, 11, 12),
    (2, 6, 10, 11, 12), (5, 7, 10, 11, 12), (6, 7, 10, 11, 12), (3, 5, 6, 8, 13), (1, 2, 7, 8, 13),
    (2, 4, 6, 9, 13), (1, 3, 4, 10, 13), (5, 7, 9, 10, 13), (1, 2, 4, 11, 13), (1, 3, 4, 11, 13),
    (2, 4, 6, 11, 13), (3, 4, 6, 11, 13), (1, 2, 8, 11, 13), (1, 3, 8, 11, 13), (2, 6, 8, 11, 13),
    (3, 6, 8, 11, 13), (1, 3, 8, 12, 13), (3, 5, 8, 12, 13), (1, 7, 8, 12, 13), (5, 7, 8, 12, 13),
    (1, 3, 10, 12, 13), (3, 5, 10, 12, 13), (1, 7, 10, 12, 13), (5, 7, 10, 12, 13), (1, 2, 4, 5, 14),
    (3, 5, 6, 8, 14), (2, 3, 7, 9, 14), (1, 6, 7, 10, 14), (4, 8, 9, 10, 14), (3, 6, 7, 11, 14),
    (3, 6, 8, 11, 14), (3, 7, 9, 11, 14), (3, 8, 9, 11, 14), (6, 7, 10, 11, 14), (6, 8, 10, 11, 14),
    (7, 9, 10, 11, 14), (8, 9, 10, 11, 14), (2, 3, 5, 12, 14), (2, 4, 5, 12, 14), (3, 5, 8, 12, 14),
    (4, 5, 8, 12, 14), (2, 3, 9, 12, 14), (2, 4, 9, 12, 14), (3, 8, 9, 12, 14), (4, 8, 9, 12, 14),
    (1, 2, 4, 13, 14), (1, 2, 7, 13, 14), (2, 4, 9, 13, 14), (2, 7, 9, 13, 14), (1, 4, 10, 13, 14),
    (1, 7, 10, 13, 14), (4, 9, 10, 13, 14), (7, 9, 10, 13, 14), (3, 4, 6, 7, 15), (1, 2, 7, 8, 15),
    (1, 5, 6, 9, 15), (2, 3, 5, 10, 15), (4, 8, 9, 10, 15), (1, 2, 5, 11, 15), (1, 2, 8, 11, 15),
    (1, 5, 9, 11, 15), (1, 8, 9, 11, 15), (2, 5, 10, 11, 15), (2, 8, 10, 11, 15), (5, 9, 10, 11, 15),
    (8, 9, 10, 11, 15), (1, 6, 7, 12, 15), (4, 6, 7, 12, 15), (1, 7, 8, 12, 15), (4, 7, 8, 12, 15),
    (1, 6, 9, 12, 15), (4, 6, 9, 12, 15), (1, 8, 9, 12, 15), (4, 8, 9, 12, 15), (3, 4, 6, 13, 15),
    (3, 5, 6, 13, 15), (4, 6, 9, 13, 15), (5, 6, 9, 13, 15), (3, 4, 10, 13, 15), (3, 5, 10, 13, 15),
    (4, 9, 10, 13, 15), (5, 9, 10, 13, 15), (1, 2, 5, 14, 15), (2, 3, 5, 14, 15), (1, 5, 6, 14, 15),
    (3, 5, 6, 14, 15), (1, 2, 7, 14, 15), (2, 3, 7, 14, 15), (1, 6, 7, 14, 15), (3, 6, 7, 14, 15),
    (4, 5, 7, 8, 16), (1, 5, 6, 9, 16), (2, 3, 7, 9, 16), (1, 3, 4, 10, 16), (2, 6, 8, 10, 16),
    (1, 3, 4, 11, 16), (1, 4, 5, 11, 16), (3, 4, 7, 11, 16), (4, 5, 7, 11, 16), (1, 3, 9, 11, 16),
    (1, 5, 9, 11, 16), (3, 7, 9, 11, 16), (5, 7, 9, 11, 16), (1, 3, 9, 12, 16), (2, 3, 9, 12, 16),
    (1, 6, 9, 12, 16), (2, 6, 9, 12, 16), (1, 3, 10, 12, 16), (2, 3, 10, 12, 16), (1, 6, 10, 12, 16),
    (2, 6, 10, 12, 16), (2, 6, 8, 13, 16), (5, 6, 8, 13, 16), (2, 7, 8, 13, 16), (5, 7, 8, 13, 16),
    (2, 6, 9, 13, 16), (5, 6, 9, 13, 16), (2, 7, 9, 13, 16), (5, 7, 9, 13, 16), (1, 4, 5, 14, 16),
    (1, 5, 6, 14, 16), (4, 5, 8, 14, 16), (5, 6, 8, 14, 16), (1, 4, 10, 14, 16), (1, 6, 10, 14, 16),
    (4, 8, 10, 14, 16), (6, 8, 10, 14, 16), (2, 3, 7, 15, 16), (3, 4, 7, 15, 16), (2, 7, 8, 15, 16),
    (4, 7, 8, 15, 16), (2, 3, 10, 15, 16), (3, 4, 10, 15, 16), (2, 8, 10, 15, 16), (4, 8, 10, 15, 16)
)

### CP2 ###

top_cp2 = (
    (1, 2, 4, 5, 6), (2, 3, 5, 6, 4), (3, 1, 6, 4, 5),
    (1, 2, 4, 5, 9), (2, 3, 5, 6, 7), (3, 1, 6, 4, 8),
    (2, 3, 6, 4, 9), (3, 1, 4, 5, 7), (1, 2, 5, 6, 8),
    (3, 1, 5, 6, 9), (1, 2, 6, 4, 7), (2, 3, 4, 5, 8),
    (4, 5, 7, 8, 9), (5, 6, 8, 9, 7), (6, 4, 9, 7, 8),
    (4, 5, 7, 8, 3), (5, 6, 8, 9, 1), (6, 4, 9, 7, 2),
    (5, 6, 9, 7, 3), (6, 4, 7, 8, 1), (4, 5, 8, 9, 2),
    (6, 4, 8, 9, 3), (4, 5, 9, 7, 1), (5, 6, 7, 8, 2),
    (7, 8, 1, 2, 3), (8, 9, 2, 3, 1), (9, 7, 3, 1, 2),
    (7, 8, 1, 2, 6), (8, 9, 2, 3, 4), (9, 7, 3, 1, 5),
    (8, 9, 3, 1, 6), (9, 7, 1, 2, 4), (7, 8, 2, 3, 5),
    (9, 7, 2, 3, 6), (7, 8, 3, 1, 4), (8, 9, 1, 2, 5)
)


rp2 = make_filtration(make_simplices(top_rp2))
rp3 = make_filtration(make_simplices(top_rp3))
rp4 = make_filtration(make_simplices(top_rp4))
cp2 = make_filtration(make_simplices(top_cp2))
