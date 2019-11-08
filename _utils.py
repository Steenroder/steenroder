import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

def _pivot(column):
    try:
        return max(column.nonzero()[0])
    except ValueError:
        return None

def get_reduced_triangular(matrix):
    '''R = MV'''
    n = matrix.shape[1]
    reduced = np.array(matrix)
    triangular = np.eye(n, dtype=np.bool)
    for j in range(n):
        i = j
        while i > 0:
            i -= 1
            if not np.any(reduced[:,j]):
                break
            else:
                piv_j = _pivot(reduced[:,j])
                piv_i = _pivot(reduced[:,i])
                
                if piv_i == piv_j:
                    reduced[:,j] = np.logical_xor(reduced[:,i], reduced[:,j])
                    triangular[:,j] = np.logical_xor(triangular[:,i], triangular[:,j])
                    i = j
                    
    return reduced, triangular

def check_factorization(matrix):
    reduced, triangular = get_reduced_triangular(matrix)
    test = np.matmul(coboundary.astype(np.int8), 
                     triangular.astype(np.int8))
    test = test%2
    return np.all(reduced == test)

def get_barcode(reduced, filtration):
    triples = []
    all_indices = []
    for j in range(len(filtration)):
        if np.any(reduced[:,j]):
            i = _pivot(reduced[:,j])
            triples.append((i,j))
            all_indices += [i,j]
    
    for i in [i for i in range(len(filtration)) if i not in all_indices]:    
        if not np.any(reduced[:,i]):
            triples.append((i,np.inf))
    
    barcode = sorted([bar for bar in triples if bar[1]-bar[0]>1])
    
    return barcode

def get_boundary(filtration):
    spx_filtration_idx = {tuple(v): idx for idx, v in enumerate(filtration)}
    boundary = np.zeros((len(filtration), len(filtration)), dtype=np.bool)
    for idx, spx in enumerate(filtration):
        faces_idxs = []
        try:
            faces_idxs = [spx_filtration_idx[spx[:j]+spx[j+1:]] 
                          for j in range(len(spx))]
        except KeyError:
            pass
        boundary[faces_idxs,idx] = True
    
    return boundary

def checking_against_gudhi(filtration):
    
    from gudhi import SimplexTree

    boundary = get_boundary(filtration)
    reduced, triangular = get_reduced_triangular(boundary)
    barcode = get_barcode(reduced, filtration)
    dimensions = [len(spx)-1 for spx in filtration]
    barcode_w_dimensions = sorted(
                            [(float(bar[0]), float(bar[1]), dimensions[bar[0]])
                            for bar in barcode])
    
    st = SimplexTree()
    for idx, spx in enumerate(filtration):
        st.insert(spx,idx)
        
    gudhi_barcode = sorted([(bar[1][0], bar[1][1], bar[0]) 
                               for bar in st.persistence(homology_coeff_field=2) 
                               if bar[1][1]-bar[1][0]>1])
    
    return [(a == b) for a,b in zip(barcode_w_dimensions, gudhi_barcode)]

def get_coboundary(filtration):
    coboundary = np.flip(get_boundary(filtration), axis=[0,1]).transpose()
    return coboundary

def get_coho_reps(barcode, reduced, triangular, filtration):
    coho_reps = np.empty((len(filtration), len(barcode)),dtype=np.bool)
    for col, pair in enumerate(barcode):
        if pair[1] < np.inf:
            coho_reps[:, col] = reduced[:,pair[1]]
        if pair[1] == np.inf:
            coho_reps[:, col] = triangular[:,pair[0]]
    return coho_reps

def check_representatives(coboundary, reps):
    test = np.matmul(coboundary.astype(np.int8), 
                     reps.astype(np.int8))
    test = test%2
    return not np.any(test)

def vector_to_cochain(vector, filtration):
    cocycle = {filtration[len(filtration)-i-1] for i in vector.nonzero()[0]}
    return cocycle

def cochain_to_vector(cochain, filtration):
    """returns a column vector shape (n,1)"""
    simplex_to_index = lambda spx: len(filtration)-filtration.index(spx)-1
    nonzero_indices = [simplex_to_index(spx) for spx in cochain]
    vector = np.zeros(shape=(len(filtration),1), dtype=np.bool)
    vector[nonzero_indices] = True
    return vector

def check_vector_to_cochain(filtration, vector):
    new_vector = cochain_to_vector(vector_to_cochain(vector, filtration), filtration)
    return np.all(vector == new_vector)

def check_cochain_to_vector(filtration, cochain):
    new_cochain = vector_to_cochain(cochain_to_vector(cochain, filtration), filtration)
    return cochain == new_cochain

def check_duality(filtration):
    boundary = get_boundary(filtration)
    barcode = get_barcode(get_reduced_triangular(boundary)[0], filtration)
    
    coboundary = get_coboundary(filtration)
    cobarcode = get_barcode(get_reduced_triangular(coboundary)[0], filtration)

    new_barcode = []
    for bar in cobarcode:
        if bar[1] == np.inf:
            new_barcode.append((len(filtration)-bar[0]-1, np.inf))
        else:
            new_barcode.append((len(filtration)-bar[1]-1, len(filtration)-bar[0]-1))
            
    return [(bar1 == bar2) for bar1, bar2 in zip(sorted(barcode), sorted(new_barcode))]

def STSQ(k, vector, filtration):
    
    # from vector to cochain
    cocycle = vector_to_cochain(vector, filtration)
    
    # bulk of the algorithm
    answer = set()
    for pair in combinations(cocycle, 2):
        a, b = set(pair[0]), set(pair[1])
        if ( len(a.union(b)) == len(a)+k and 
        tuple(sorted(a.union(b))) in filtration ):
            a_bar, b_bar = a.difference(b), b.difference(a)
            index = dict()
            for v in a_bar.union(b_bar):
                pos = sorted(a.union(b)).index(v)
                pos_bar = sorted(a_bar.union(b_bar)).index(v)
                index[v] = (pos + pos_bar)%2
            index_a = {index[v] for v in a_bar}
            index_b = {index[w] for w in b_bar}
            if (index_a == {0} and index_b == {1} 
            or  index_a == {1} and index_b == {0}):
                u = sorted(a.union(b))
                answer ^= {tuple(u)}
    
    # cochain to vector
    st_rep = cochain_to_vector(answer, filtration)
    
    return st_rep

def get_steenrod_reps(k, coho_reps, filtration):
    steenrod_reps = np.empty(coho_reps.shape ,dtype=np.bool)
    for idx, rep in enumerate(np.transpose(coho_reps)):
        steenrod_reps[:,idx:idx+1] = STSQ(k,rep,filtration)
    return steenrod_reps

def betti_curves(barcode, filtration):
    dim = max([len(spx)-1 for spx in filtration])
    betti_curves = {i: np.zeros((len(filtration),), np.int8) 
                   for i in range(dim+1)}
    for bar in barcode:        
        degree = len(filtration[-bar[0]-1])-1
        end = bar[1]
        if end == np.inf:
            end = len(filtration)
            
        betti_curves[degree][bar[0]:end] += 1
            
    return betti_curves

def get_pivots(matrix):
    n = matrix.shape[1]
    pivots = []
    for i in range(n):
        pivots.append(_pivot(matrix[:,i]))
    return pivots

def reduce_vector(reduced, vector):
    num_col = reduced.shape[1]
    i = -1
    while i >= -num_col:
        if not np.any(vector):
            break
        else:
            piv_v = _pivot(vector)
            piv_i = _pivot(reduced[:,i])

            if piv_i == piv_v:
                vector[:,0] = np.logical_xor(reduced[:,i], vector[:,0])
                i = 0
            i -= 1
    return vector

def reduce_matrix(reduced, matrix):
    num_vector = matrix.shape[1]

    for i in range(num_vector):
        reduced_vector = reduce_vector(reduced, matrix[:, i:i+1])
        reduced = np.concatenate([reduced, reduced_vector], axis=1)
    return reduced[:, -num_vector:]

def get_rank(matrix):
    sums = np.sum(matrix,axis=0)
    rank = len(sums.nonzero()[0])
    return rank

def steenrod_curve(barcode, steenrod_reps, filtration, reduced):
    steenrod_matrix = np.array(steenrod_reps)
    births = [pair[0] for pair in barcode] + [len(filtration)]
    
    curve = [0]*births[0]
    for i, b in enumerate(births[:-1]):
        for j in range(b, births[i+1]):
            reducing = np.hstack((reduced[:,:j], steenrod_matrix[:,:i]))
            steenrod_matrix[:,i:i+1] = reduce_vector(reducing, steenrod_matrix[:,i:i+1])
            curve.append(get_rank(steenrod_matrix[:,:i+1]))

    return curve

