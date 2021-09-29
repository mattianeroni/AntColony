import aco
import tsp

import random


def random_walk (dists, length=25):
    """
    Given a matrix of distances, this method returns a path of the required length.
    """
    options = list(range(1, len(dists)))
    return random.sample(options, length)


dists = tsp.distance_matrix
path = random_walk(dists, 10)

a = aco.AntColony(dists, path, max_wu=400)
_, cost_wu = a.run()

a = aco.AntColony(dists, path, max_wu=0)
_, cost = a.run()

print(cost, cost_wu)
