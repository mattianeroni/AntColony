import warehouse
from tsp import TSP
from aco import AntColony

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



for i in range(5):
    problem = TSP(nodes = 60, space_size = (1000, 1000))
    myaco =
    for size in range(20, 70, 10):
        tour = random_walk(problem.distance_matrix, size)




#a = aco.AntColony(dists, path, warmup="bellaachia")
#print(a.pheromone)
