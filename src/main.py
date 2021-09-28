import aco
import tsp


a = aco.AntColony(tsp.distance_matrix, [1,2,4])

print(a.pheromone)
