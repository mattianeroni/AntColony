import warehouse
from tsp import TSP
from aco import AntColony

import random
import statistics as stats

#import sys
#import numpy as np
#np.set_printoptions(threshold=sys.maxsize)


def random_walk (dists, length=25):
    """
    Given a matrix of distances, this method returns a path of the required length.
    """
    options = list(range(1, len(dists)))
    return random.sample(options, length)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#           TESTS ON GENERIC TRAVELLING SALESMAN PROBLEM INSTANCES
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

results_dict = dict()

for i in range(5):
    # For each graph (5 different graphs)

    # Log and prepare disctionary of results
    results_dict[i] = dict()
    print("Graph : ", i)

    # Build a problem
    problem = TSP(nodes = 61, space_size = (1000, 1000))
    dists = problem.distance_matrix

    # Runs the warmup according to the given matrix of distances.
    algorithms = [
        AntColony(dists, warmup = "none"),
        AntColony(dists, warmup = "Mattia"),
        AntColony(dists, warmup = "Dai"),
        AntColony(dists, warmup = "Bellaachia")
    ]

    for size in range(20, 70, 10):
        # Log and prepare disctionary of results
        results_dict[i][size] = dict()
        print("Problem complexity : ", size)

        # Define a tour of a different complexity
        tour = random_walk(dists, size)

        for algorithm in algorithms:
            print("Algorithm : ", algorithm.warmup, end = " - ")
            costs, computations, times = [], [], []
            for j in range(5):
                # Log the iteration
                print(j, end=", ")
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

            print("\n", end="")
            # Save the results in a dictionary
            results_dict[i][size][algorithm.warmup] = {
                "Cost" : (int(stats.mean(costs)), int(stats.stdev(costs))),
                "Iterations" : (int(stats.mean(computations)), int(stats.stdev(computations))),
                "Time" : (round(stats.mean(times), 3), round(stats.stdev(times), 3)),
            }

# Print the dictionary of results
print(results_dict)

# Export results_dict to .csv file
with open("../results/costs_experiments.csv", "w") as file:
    # Costs
    file.write("Costs \n")
    file.write("G, N, ACO, ACO, ACOWU, ACOWU, Dai, Dai, Bellaachia, Bellaachia,\n")
    file.write(",, Avg., St.Dev., Avg., St.Dev., Avg., St.Dev., Avg., St.Dev.,\n")
    for i, graph in results_dict.items():
        for size, experiment in graph.items():
            file.write(f"{i}, {size}, ")
            for _, algorithm in experiment.items():
                avg, stdev = algorithm["Cost"][0], algorithm["Cost"][1]
                file.write(f"{avg}, {stdev}, ")
            file.write("\n")

with open("../results/iterations_experiments.csv", "w") as file:
    # Iterations needed to reach the best
    file.write("Iterations needed \n")
    file.write("G, N, ACO, ACO, ACOWU, ACOWU, Dai, Dai, Bellaachia, Bellaachia,\n")
    file.write(",, Avg., St.Dev., Avg., St.Dev., Avg., St.Dev., Avg., St.Dev.,\n")
    for i, graph in results_dict.items():
        for size, experiment in graph.items():
            file.write(f"{i}, {size}, ")
            for _, algorithm in experiment.items():
                avg, stdev = algorithm["Iterations"][0], algorithm["Iterations"][1]
                file.write(f"{avg}, {stdev}, ")
            file.write("\n")

with open("../results/times_experiments.csv", "w") as file:
    # Iterations needed to reach the best
    file.write("Time \n")
    file.write("G, N, ACO, ACO, ACOWU, ACOWU, Dai, Dai, Bellaachia, Bellaachia,\n")
    file.write(",, Avg., St.Dev., Avg., St.Dev., Avg., St.Dev., Avg., St.Dev.,\n")
    for i, graph in results_dict.items():
        for size, experiment in graph.items():
            file.write(f"{i}, {size}, ")
            for _, algorithm in experiment.items():
                avg, stdev = str(algorithm["Time"][0]), str(algorithm["Time"][1])
                file.write(f"{avg}, {stdev}, ")
            file.write("\n")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                    TESTS ON THE REALISTIC WAREHOUSE LAYOUT
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

results_dict = dict()
dists = warehouse.distance_matrix

# Runs the warmup according to the given matrix of distances.
algorithms = [
    AntColony(dists, warmup = "none"),
    AntColony(dists, warmup = "Mattia"),
    AntColony(dists, warmup = "Dai"),
    AntColony(dists, warmup = "Bellaachia")
]

for size in range(20, 70, 10):
    # Log and prepare disctionary of results
    results_dict[size] = dict()
    print("Problem complexity : ", size)

    # Define a tour of a different complexity
    tour = random_walk(dists, size)

    for algorithm in algorithms:
        print("Algorithm : ", algorithm.warmup, end = " - ")
        costs, computations, times = [], [], []
        for j in range(5):
            # Log the iteration
            print(j, end=", ")
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

        print("\n", end="")
        # Save the results in a dictionary
        results_dict[size][algorithm.warmup] = {
            "Cost" : (int(stats.mean(costs)), int(stats.stdev(costs))),
            "Iterations" : (int(stats.mean(computations)), int(stats.stdev(computations))),
            "Time" : (round(stats.mean(times), 3), round(stats.stdev(times), 3)),
        }

# Print the dictionary of results
print(results_dict)

# Export results_dict to .csv file
with open("../results/costs_warehouse.csv", "w") as file:
    # Costs
    file.write("Costs \n")
    file.write("N, ACO, ACO, ACOWU, ACOWU, Dai, Dai, Bellaachia, Bellaachia,\n")
    file.write(", Avg., St.Dev., Avg., St.Dev., Avg., St.Dev., Avg., St.Dev.,\n")
    for size, experiment in results_dict.items():
        file.write(f"{size}, ")
        for _, algorithm in experiment.items():
            avg, stdev = algorithm["Cost"][0], algorithm["Cost"][1]
            file.write(f"{avg}, {stdev}, ")
        file.write("\n")

with open("../results/iterations_warehouse.csv", "w") as file:
    # Iterations needed to reach the best
    file.write("Iterations needed \n")
    file.write("N, ACO, ACO, ACOWU, ACOWU, Dai, Dai, Bellaachia, Bellaachia,\n")
    file.write(", Avg., St.Dev., Avg., St.Dev., Avg., St.Dev., Avg., St.Dev.,\n")
    for size, experiment in results_dict.items():
        file.write(f"{size}, ")
        for _, algorithm in experiment.items():
            avg, stdev = algorithm["Iterations"][0], algorithm["Iterations"][1]
            file.write(f"{avg}, {stdev}, ")
        file.write("\n")

with open("../results/times_warehouse.csv", "w") as file:
    # Iterations needed to reach the best
    file.write("Time \n")
    file.write("N, ACO, ACO, ACOWU, ACOWU, Dai, Dai, Bellaachia, Bellaachia,\n")
    file.write(", Avg., St.Dev., Avg., St.Dev., Avg., St.Dev., Avg., St.Dev.,\n")
    for size, experiment in results_dict.items():
        file.write(f"{size}, ")
        for _, algorithm in experiment.items():
            avg, stdev = str(algorithm["Time"][0]), str(algorithm["Time"][1])
            file.write(f"{avg}, {stdev}, ")
        file.write("\n")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
