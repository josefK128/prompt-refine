# knn_urls - k nearest neighbor urls
# finds the K nearest vectors to L[0] (prompt vector) among L[1] to L[N-1].
# returns the K urld corresponding to the nearest vectors

from typing import List
from random import sample
#from vector import Vector
#from euclidean_distance import euclidean_distance
from cosine_similarity import CosineSimilarity
import numpy as np



def action(W:np.ndarray, urls: List[str], K: int) -> List[str]:
    
    print(f'\n\n%%%%% knn_urls.action: W.shape = {W.shape}')
    nrows, ncols = W.shape

    if nrows <= 1:
        raise ValueError("List L must contain at least 2 vectors")
    if K >= nrows:
        return urls[1:]



    # form residual matrix when prompt target_vector (row0) is removed
    target_vector = W[0, :]
    D = np.delete(W , (0), axis=0)
    print(f'\n\nD.shape = {D.shape}')


    # find list of distances from rows of D from target_vector
    distances = []
    index = 1
    for row in D:
        print(f'row = {row}')
        distance = CosineSimilarity.action(target_vector, row)
        print(f'distance = {distance}')
        distances.append((np.absolute(distance), urls[index]))
        index += 1

#    target_vector = Vector(L[0])
#    distances = []
#    for i in range(1, len(L)):
#        vector = Vector(L[i])
#        print(f'vector = {vector}')
#        distance = euclidean_distance(target_vector, vector)
#        distances.append((distance, urls[i]))

    # prompt vector - embedding of prompt - first row of matrix U = L[0] 
    # NOTE: distances is a List of Tuples [d, urls[i]]  
#    prompt_vector = L[0]
#    distances = []
#    for i in range(1, len(L)):
#        delta = [0] * len(L)
#        for j in range(len(delta)):
#            delta[j] += L[i][j] -  L[0][j]
#        print(f'delta = {delta}')
#        d = np.linalg.norm(delta)
#        print(f'd = {d}')
#        distances.append((d, urls[i]))

    
    print('\n')
    distances.sort(key=lambda x: x[0])
    for d, url in distances:
        print(f'sorted distance d = {d} with url = {url}')
    

    nearest_neighbors = []
    for distance, url in distances:
        if len(nearest_neighbors) >= K:
            break
        nearest_neighbors.append(url)
    

    if len(nearest_neighbors) < K:
        remaining_urls = [url for _, url in distances[K - 1:]]
        nearest_neighbors.extend(sample(remaining_urls, K - len(nearest_neighbors)))
    
    return nearest_neighbors

