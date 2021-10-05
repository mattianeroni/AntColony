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


results_dict = dict()

for i in range(5):
    # For each graph (5 different graphs)

    # Build a problem
    problem = TSP(nodes = 60, space_size = (1000, 1000))
    dists = problem.distance_matrix

    # Runs the warmup according to the given matrix of distances.
    algorithms = [
        AntColony(dists, warmup = "none"),
        AntColony(dists, warmup = "Mattia").
        AntColony(dists, warmup = "Dai"),
        AntColony(dists, warmup = "Bellaachia")
    ]

    for size in range(20, 70, 10):
        # Define a tour of a different complexity
        tour = random_walk(problem.distance_matrix, size)

        for algorithm in algorithms:
            costs, computations, times = [], [], []
            for j in range(3):
                # Run the algorithm
                algorithm.run(tour)
                # Save the results obtained over three executions in terms of
                # cost of the best, iterations needed to find the best, computational time.
                costs.append(algorithm.vbest)
                computations.append(algorithm.computations)
                times.append(algorithm.computational_time)
                # Reset the parameters for next execution and restore the
                # pheromone as after the warmup
                algorithm.reset()
