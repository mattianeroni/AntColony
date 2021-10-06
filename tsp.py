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
import matplotlib.pyplot as plt

# Default parameters:
# NODES = 30: The number of nodes in the graph.
# SPACE_SIZE = (1000,1000) : The size of the area in which these nodes are placed.

def euclidean (x, y):
    """
    The euclidean distance between two coordinates expressed
    as two tuples.
    """
    return int(math.sqrt( (x[0] - y[0])**2 + (x[1] - y[1])**2 ))



class TSP (object):
    """
    An instance of this class represents a gereric graph for
    Travelling Salesman Problem.
    """

    def __init__ (self, nodes = 30, space_size = (1000, 1000)):
        # The graph instance
        G = nx.Graph()

        # The nodes list
        nodes_dict = dict()

        # Create nodes
        for i in range(nodes):
            nodes_dict[i] = (random.randint(0, space_size[0]), random.randint(0, space_size[1]))
            G.add_node(i)

        # Create edges
        for i, j in itertools.permutations(range(nodes), 2):
            G.add_edge(i, j, weight=euclidean(nodes_dict[i], nodes_dict[j]))

        self.G = G
        self.nodes = nodes_dict
        self.distance_matrix = nxalg.floyd_warshall_numpy(G)


    def plot (self):
        """
        This method plot the generated graph.
        """
        nx.draw(self.G, pos=self.nodes, with_labels=True, font_weight='bold')
        plt.show()
