import random
import numpy as np
import time
import operator
import collections
import itertools

from mst import MinimalSpanningTree


def _compute_distance (lst, distances):
    """
    Given a picking list and a distance matrix, this method calculates the
    distance ran to complete the picking list.

    :param lst: The picking list
    :param distances: The distance matrix
    :return: The distance ran.
    """
    return sum(distances[lst[i]][lst[i+1]] for i in range(len(lst) - 1)) + distances[lst[-1]][0] + distances[0][lst[0]]



class AntColony (object):
    """
    This is the Ant Colony Optimization algorithm with Warm-Up.
    """
    def __init__ (self, distances,
                pher_init = 0.1, ro = 0.5, Q = 100.0, alpha = 1.0, beta = 2.0,
                evaporate = False, max_iter = 3000, max_noimp = 1000, print_every = 100,
                warmup = "mattia", max_wu = 300, ro_wu = 1.0 ):
        """
        Initialize.

        :attr distances: The distance matrix.
        :attr ro: A parameter that defines the evaporation of the pheromone.
        :attr Q: A parameter that defines the increment of the pheromone on
                the new best path.
        :attr alpha, beta: Parameters of the empirical distribution used to
                            select the next node at each step.
        :attr evaporate: If TRUE the pheromone evaporates at every iteration,
                        otherwise only when a better best solution is found.
        :attr max_iter: The number of iterations.
        :attr max_noimp: Maximum number of iterations without improvement.
        :attr print_every: The iterations between a log and the next one.

        :attr warmup: The initialisation algorithm.
        :attr max_wu: Maximum number of iterations for the Warm-Up process.
        :attr ro_wu: The value of ro used during the Warm-Up.

        :attr pheromone: The pheromone on each arch.
        :attr best: The best solution found so far.
        :attr vbest: the cost of the current best.
        :attr history: The history of the best solutions found.
        :attr computations: The number of solutions explored before finding the best.

        :param pher_init: The initial pheromone (it is always 0 on arcs (i,j) where i == j).

        """
        self.distances = distances
        self.picking_list = None    # This is set when method run is called
        self.ro = ro
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.evaporate = evaporate
        self.max_iter = max_iter
        self.max_noimp = max_noimp
        self.print_every = print_every
        self.pher_init = pher_init
        self.warmup = warmup
        self.max_wu = max_wu
        self.ro_wu = ro_wu

        # Initialize the best
        self.best = list(self.picking_list)
        random.shuffle(self.best)
        self.vbest = _compute_distance (self.best, distances)

        # Initialize the pheromone
        self.pheromone = np.full(distances.shape, pher_init)
        np.fill_diagonal(self.pheromone, 0)
        # Eventual warmup procedure
        self.warmup_process(warmup)

        # Save the state of pheromone after warmup
        # to avoid recalculating it.
        self.saved_pheromone = self.pheromone.copy()

        # Initialize the history and the number of iterations needed to find the best
        # and other statistics.
        self.history = collections.deque((self.vbest,))
        self.computations = 0
        self.computational_time = 0.0


    def warmup_process (self, warmup):
        """
        The warmup procedure to initialise the pheromone matrix.
        """
        alpha, beta, Q, ro_wu = self.alpha, self.beta, self.Q, self.ro_wu
        distances = self.distances
        if warmup == "Mattia":
            for _ in range(max_wu):
                C = distances + np.identity(distances.shape[0])
                desirability = self.pheromone**alpha * (1 / C)**beta
                P = desirability / desirability.sum(axis=1)
                U = Q / C
                self.pheromone += U * P
                self.pheromone *= ro_wu
        elif warmup == "Dai":
            minimal_spanning_tree = MinimalSpanningTree(distances)
            for i, j in minimal_spanning_tree.edges:
                self.pheromone[i, j] **= 1 / beta
                self.pheromone[j, i] **= 1 / beta
        elif warmup == "Bellaachia":
            for i, j in itertools.combinations(range(distances.shape[0]), 2):
                self.pheromone[i, j] = 1 / (distances[i,:].sum() - distances[i, j])
                self.pheromone[j, i] = 1 / (distances[j,:].sum() - distances[j, i])

        elif warmup == "none":
            pass

        else:
            raise Exception("The warmup required doesn't exist.")


    def reset (self):
        """
        This method resets the algorithm.
        """
        # Initialize the best
        self.best = list(self.picking_list)
        random.shuffle(self.best)
        self.vbest = _compute_distance (self.best, distances)

        # Initialize the pheromone
        self.pheromone = self.saved_pheromone.copy()

        # Initialize the history and the number of iterations needed to find the best
        # and other statistics.
        self.history = collections.deque((self.vbest,))
        self.computations = 0
        self.computational_time = 0.0



    def _evap (self):
        """
        This method evaporates the pheromone.
        """
        self.pheromone *= self.ro


    def _update (self):
        """
        This method updates the pheromone on the best path.
        """
        for i in range (len(self.picking_list) - 1):
            self.pheromone[self.best[i], self.best[i + 1]] += (self.Q / self.distances[self.best[i], self.best[i + 1]])
        self.pheromone[0, self.best[0]] += (self.Q / self.distances[0, self.best[0]])
        self.pheromone[self.best[-1], 0] += (self.Q / self.distances[self.best[-1], 0])


    def _next_node (self, options):
        """
        This method returns the next node during the constructing process that
        brings to a new solution.
        The node is selected in a list of possible <options>. Each option is given
        by a tuple containing (the node, its desirability).
        Given i the current node, the desirability of node j (i.e. d(j)) is
        calculated as follows:

        d(j) = ph(i,j)^alpha / dist(i,j)^beta

        where alpha and beta are parameters of the algorithm, dist(i,j) is the
        distance from i to j, and ph(i,j) is the pheromone on the edge (i,j).

        The probability to select a node is calculated dividing its desirability
        for the total desirability of all the options.

        :param options: List of tuples (node, desirability of node).
        :return: The selected node.

        """
        prob = 0.0
        r = random.random()
        options.sort(key=operator.itemgetter(1), reverse=True)
        total = sum(desirability for _, desirability in options)

        for op, desirability in options:
            prob += desirability / total
            if r < prob:
                return op
        return -1


    def _new_solution (self):
        """
        This method construct node by node a new solution.
        """
        c_node = 0
        new_sol, vnew_sol = [], 0
        options = list(self.picking_list)

        for i in range (len(self.picking_list)):
            options_params = [(op, self.pheromone[c_node, op]**self.alpha / self.distances[c_node, op]**self.beta) for op in options]
            n_node = self._next_node (options_params)
            new_sol.append(n_node)
            options.remove (n_node)
            vnew_sol += self.distances[c_node, n_node]
            c_node = n_node
        vnew_sol += self.distances[c_node, 0]

        return new_sol, vnew_sol


    def run (self, picking_list, verbose = False):
        """
        This method represents the execution of the algorithm.

        :param picking_list: The tour of nodes for which the problem must be solved.
        :param verbose: If TRUE a log takes place every <print_every> iterations.
        :return: The best solution and its cost.

        """
        self.picking_list = list(picking_list)
        start = time.time()
        noimp = 0
        for i in range (self.max_iter):
            # Build a new solution
            new_sol, vnew_sol = self._new_solution ()
            # Eventually evaporate pheromone
            if self.evaporate is True:
                self._evap ()
            # Eventually update best, iterations with no improvement
            # and computations needed to find the best.
            if vnew_sol < self.vbest:
                self.best, self.vbest = new_sol, vnew_sol
                if self.evaporate is False:
                    self._evap ()
                self._update ()
                noimp = 0
                self.computations = i
            else:
                noimp += 1
                if noimp > self.max_noimp:
                    break

            # Update history
            self.history.append(self.vbest)
            # Logs
            if verbose is True and i % self.print_every == 0:
                print('Epoch: ', i, ', Best: ', self.vbest)

        # Set computational time
        self.computational_time = time.time() - start
        # Return the best solution found
        return self.best, self.vbest
