import aco
import tsp
import mst
import warehouse

import random

#import sys
#import numpy as np
#np.set_printoptions(threshold=sys.maxsize)


def random_walk (dists, length=25):
    """
    Given a matrix of distances, this method returns a path of the required length.
    """
    options = list(range(1, len(dists)))
    return random.sample(options, length)


dists = tsp.distance_matrix
path = random_walk(dists, 10)


a = aco.AntColony(dists, path, warmup="bellaachia")
print(a.pheromone)
