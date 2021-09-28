"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This file contains the implementation of a generic graph for the
Travelling Salesman Problem (TSP).

Author: Mattia Neroni, Ph.D., Eng. (Set 2021).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import networkx as nx
import networkx.algorithms.shortest_paths.dense as nxalg
import random
import itertools
import math
#import matplotlib.pyplot as plt

# Parameters:
# NODES : The number of nodes in the graph.
# SPACE_SIZE : The size of the area in which these nodes are placed.
NODES = 30
SPACE_SIZE = (1000, 1000)


def euclidean (x, y):
    """
    The euclidean distance between two coordinates expressed
    as two tuples.
    """
    return int(math.sqrt( (x[0] - y[0])**2 + (x[1] - y[1])**2 ))


# The graph instance
G = nx.Graph()

# The nodes list
nodes = dict()

# Create nodes
for i in range(NODES):
    nodes[i] = (random.randint(0, SPACE_SIZE[0]), random.randint(0, SPACE_SIZE[1]))
    G.add_node(i)

# Create edges
for i, j in itertools.permutations(range(NODES), 2):
    G.add_edge(i, j, weight=euclidean(nodes[i], nodes[j]))


# Plot the graph
#nx.draw(G, pos=nodes, with_labels=True, font_weight='bold')
#plt.show()

# Set the distance matrix
distance_matrix = nxalg.floyd_warshall_numpy(G)
#print(distance_matrix)
