"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This file contains the implementation of some algorithms to calculate the
Minimal Spanning Tree in a graph.
Currently, only the Prim's Algorithm is implemented.


Author: Mattia Neroni, Ph.D., Eng. (Set 2021).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import itertools
import collections
#import networkx as nx


class MinimalSpanningTree (object):
    """
    Implementation of the Prim's Algorithm for finding the Minimal Spanning
    Tree in a graph.
    """

    def __init__(self, dists, algorithm="prim"):
        self.dists = dists

        F = {0}
        Q = set(range(1, dists.shape[0]))
        edges = collections.deque()
        while len(Q) > 0:
            mincost, edge = dists.max(), None
            for i in F:
                for j in Q:
                    if (cost := dists[i, j]) < mincost:
                        mincost, edge = cost, (i,j)

            edges.append(edge)
            F.add(edge[1]); Q.remove(edge[1])

        self.F = F
        self.edges = edges



    def plot(self):
        pass
