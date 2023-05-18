"""

==============================================================
Digraph Creation, (:mod:`f_abm.src.digraph_creation`)
==============================================================

Description
-----------

    This module contains all the digraph creation functions

Functions
---------


"""


import random
import numpy as np
from src.auxiliary_functions import (create_random_numbers, add_rs_weights2matrix, add_signs2matrix, add_random_edges, )


def default_digraph(default_type=0, num_agents=10):
    """
    This function returns pre-made digraphs to be used primarily as default for functions. The pre-made digraph that
    will be called for default will always be the one with default_type=0
    :param default_type: ID of the default digraph
    :param num_agents: number of agents
    :return: the corresponding adjacency matrix
    """

    if default_type == 1:
        # Random digraph
        digraph = random_digraph(num_agents=num_agents, row_stochastic=True, positive_edge_ratio=1.0,
                                 edge_probability=0.8)

        return digraph
    elif default_type == 2:
        # Ring digraph
        digraph = ring_digraph(num_agents=num_agents, row_stochastic=True, positive_edge_ratio=1.0)

        return digraph
    elif default_type == 0:
        # Small-world
        random_parameters = [[0, -1.0, -0.7, 1], [0, -0.2, 0.2, 1], [0, 0.7, 1.0, 1]]
        random_numbers = create_random_numbers(num_agents=num_agents, number_parameters=random_parameters,
                                               limits=(0, 1))
        digraph = small_world_digraph(num_agents=num_agents, topology_signature=[0, 1, 3, -5],
                                      change_probability=random_numbers,
                                      positive_edge_ratio=0.5)

        return digraph

    else:
        digraph = ring_digraph()

        return digraph


def complete_digraph(num_agents=100, row_stochastic=False, positive_edge_ratio=1.0):
    """ This is a function that returns a complete digraph

    :param num_agents: Is the number of agents (and therefore vertices) of the digraph. By default, it is 100
    :param row_stochastic: A boolean that determines if the returned digraph must
                                have a row-stochastic matrix. By default, this is False
    :param positive_edge_ratio: A floating number between 0 and 1 that determines
                                the ratio of positive edges in the digraph. By default, it is 1
    :return: The adjacency matrix of the corresponding generalised ring digraph

    """

    # First, create the topology
    # Initialise an array of zeros
    adjacency_matrix = np.ones((num_agents, num_agents))

    # Now, if it is row-stochastic, add the weights
    if row_stochastic:
        add_rs_weights2matrix(adjacency_matrix)

    # If the matrix is signed, add the negative edges
    if positive_edge_ratio < 1:
        add_signs2matrix(adjacency_matrix, positive_edge_ratio)

    return adjacency_matrix


def ring_digraph(num_agents=100, topology_signature=None, row_stochastic=False, positive_edge_ratio=1.0,
                 num_random_edges_it=0):
    """ This is a function that returns a ring digraph

    :param num_agents: Is the number of agents (and therefore vertices) of the digraph. By default, it is 100
    :param topology_signature: Is a list with the relative indices of the
                               vertices that influence each agent. By default, it is [0, 1]
    :param row_stochastic: A boolean that determines if the returned digraph must
                                have a row-stochastic matrix. By default, this is False
    :param positive_edge_ratio: A floating number between 0 and 1 that determines
                                the ratio of positive edges in the digraph. By default, it is 1
    :param num_random_edges_it: number of iterations to add random edges
    :return: The adjacency matrix of the corresponding generalised ring digraph

    """

    # print('f = ring_digraph')

    # First, create the topology
    # Initialise an array of zeros
    adjacency_matrix = np.zeros((num_agents, num_agents))

    # If the topology_signature is None then it is a simple ring digraph
    if topology_signature is None:
        topology_signature = [0, 1]

    # All the vertices have a self-loop, if it is not included in the
    # signature, include it
    if 0 not in topology_signature:
        topology_signature = np.concatenate((topology_signature, np.array([0])))

    # Go row by row applying the topology_signature
    for id_row in range(0, num_agents):
        for relative_neighbour in topology_signature:
            absolute_neighbour = (id_row + relative_neighbour)
            while num_agents <= absolute_neighbour:
                absolute_neighbour = absolute_neighbour - num_agents
            while 0 > absolute_neighbour:
                absolute_neighbour = absolute_neighbour + num_agents
            adjacency_matrix[id_row, absolute_neighbour] = 1

    # If necessary, add random edges
    if num_random_edges_it > 0:
        add_random_edges(adjacency_matrix=adjacency_matrix, num_iterations=num_random_edges_it)

    # Now, if it is row-stochastic, add the weights
    if row_stochastic:
        add_rs_weights2matrix(adjacency_matrix)

    # If the matrix is signed, add the negative edges
    if positive_edge_ratio < 1:
        add_signs2matrix(adjacency_matrix, positive_edge_ratio)

    return adjacency_matrix


def small_world_digraph(num_agents=100, topology_signature=None, row_stochastic=False, positive_edge_ratio=1.0,
                        change_probability=0.0, reverse_probability=0.0, bidirectional_probability=0.0,
                        num_random_edges_it=0):
    """ This is a function that creates a digraph with small-world topology

    :param num_agents: number of agents, by default 100
    :param topology_signature: the topology signature of the underlying ring digraph
    :param row_stochastic: whether the adjacency matrix is row-stochastic, by default False
    :param positive_edge_ratio: the positive edge ratio, by default 1
    :param change_probability: the probability of edges changing target, it accepts a number between 0.0 and 1.0 or a
        list of 'num_agents' numbers between 0.0 and 1.0. Each element in the list corresponds to the change probability
        of the corresponding vertex
    :param reverse_probability: the probability of edges reversing target, it accepts a number between 0.0 and 1.0 or a
        list of 'num_agents' numbers between 0.0 and 1.0. Each element in the list corresponds to the reverse
        probability of the corresponding vertex
    :param bidirectional_probability: the probability of edges being bidirectional, it accepts a number between 0.0 and
        1.0 or a list of 'num_agents' numbers between 0.0 and 1.0. Each element in the list corresponds to the
        probability of the corresponding vertex being bidirectional
    :param num_random_edges_it: number of iterations to add random edges
    :return: the adjacency matrix associated with the corresponding small-world digraph
    """

    # print('f = small_world_digraph')

    # Preparation:
    # If the 'change_probability', 'reverse_probability', or 'bidirectional_probability' parameters are single numbers,
    # transform them into an array
    if type(change_probability) is float:
        change_probability = np.ones((1, num_agents))*change_probability
        change_probability = change_probability.squeeze()

    if type(reverse_probability) is float:
        reverse_probability = np.ones((1, num_agents))*reverse_probability
        reverse_probability = reverse_probability.squeeze()

    if type(bidirectional_probability) is float:
        bidirectional_probability = np.ones((1, num_agents))*bidirectional_probability
        bidirectional_probability = bidirectional_probability.squeeze()

    # First, create the corresponding Ring topology (at this moment we do not care about the signs or weights)
    adjacency_matrix = ring_digraph(num_agents=num_agents, topology_signature=topology_signature,
                                    num_random_edges_it=num_random_edges_it)

    # Now, go edge by edge and with a certain probability move it to another vertex
    # List all the non self-loop edges
    edges = [[id_row, id_col] for id_row in range(num_agents) for id_col in range(num_agents)
             if (id_row != id_col and adjacency_matrix[id_row, id_col] != 0)]

    for edge in edges:
        # Select the source vertex
        source_vertex = edge[1]

        # Select the old target vertex
        target_vertex = edge[0]

        # Only allow for an edge change if it is not the self-loop
        if source_vertex != target_vertex:

            local_change_probability = change_probability[source_vertex]
            local_bidirectional_probability = bidirectional_probability[source_vertex]
            local_reverse_probability = reverse_probability[source_vertex]

            if np.random.uniform(low=0.0, high=1.0) < local_change_probability:
                # Rewire that edge

                # Get all the vertices that the source vertex does not influence
                possible_vertices = (adjacency_matrix[:, source_vertex] == 0).nonzero()[0]

                # Get a random vertex from these available vertices
                array_length = len(possible_vertices) # :=
                if array_length > 0:
                    # If the list of possible new vertices is not empty
                    # the new vertex is one of the possible vertices chosen at random
                    new_vertex = possible_vertices[random.randint(0, array_length-1)]
                else:
                    # If the list of possible new vertices is empty, do not change
                    new_vertex = target_vertex

                # Now, modify the adjacency matrix
                # Erase the previous vertex from the adjacency matrix
                adjacency_matrix[target_vertex, source_vertex] = 0.0

                if np.random.uniform(low=0.0, high=1.0) < local_bidirectional_probability:
                    # The edge is bidirectional
                    # Add the new vertices
                    adjacency_matrix[source_vertex, new_vertex] = 1.0
                    adjacency_matrix[new_vertex, source_vertex] = 1.0

                else:
                    # The edge is not bidirectional
                    if np.random.uniform(low=0.0, high=1.0) < local_reverse_probability:
                        # The edge is reversed, reverse the edge
                        # Add the new vertex
                        adjacency_matrix[source_vertex, new_vertex] = 1.0
                    else:
                        # The edge is not reversed
                        # Add the new vertex
                        adjacency_matrix[new_vertex, source_vertex] = 1.0

    # Now that the topology is ready, add the signs and weights if necessary

    # If the digraph is row-stochastic, add the weights
    if row_stochastic:
        add_rs_weights2matrix(adjacency_matrix)

    # If the matrix is signed, add the negative edges
    if positive_edge_ratio < 1:
        add_signs2matrix(adjacency_matrix, positive_edge_ratio)

    return adjacency_matrix


def random_digraph(num_agents=100, row_stochastic=False, positive_edge_ratio=1.0, edge_probability=0.5):
    """
    This function creates a digraph with random topology. Note that not all the edges are random. The resulting
    adjacency matrix always has non-zero elements in the diagonal, indicating the self-loop

    :param num_agents: number of agents of the digraph, by default 100
    :param row_stochastic: boolean indicating if the adjacency matrix is row-stochastic
    :param positive_edge_ratio: the positive edge ratio
    :param edge_probability: the probability that an edge will exist
    :return: the adjacency matrix
    """

    # print('f = random_digraph')

    # First, create the topology
    # Initialise an identity matrix
    adjacency_matrix = np.eye(num_agents)

    num_possible_edges = num_agents*(num_agents-1)  # *0.5  # The number of possible edges, excluding self-loops
    num_edges = int(np.floor(edge_probability * num_possible_edges))  # Number of requested edges
    num_edges = np.maximum(0, num_edges-num_agents)  # Subtract the number of self-loops, and it cannot be less than 0

    # There are two methods to allocate the random edges, one is better for low probabilities
    if edge_probability < 0.4:
        # Randomly sample the adjacency matrix, if the sampled edge does not exist, make create it
        while num_edges > 0:

            # Select a random edge
            id_row = random.randint(0, num_agents - 1)
            id_col = random.randint(0, num_agents - 1)

            if adjacency_matrix[id_row][id_col] == 0.0:
                adjacency_matrix[id_row][id_col] = 1.0
                num_edges -= 1

    else:
        # List all possible edges, shuffle them and select the first 'num_edges'
        edges = [[id_row, id_col] for id_row in range(num_agents) for id_col in range(num_agents)
                 if adjacency_matrix[id_row][id_col] == 0]

        # Sort the edges randomly
        rng = np.random.default_rng()
        rng.shuffle(edges)

        # Take the first 'num_edges' ones
        edges = np.array(edges)[:num_edges, :]

        # Change add the edge to the adjacency matrix
        for id_row, id_col in edges:
            adjacency_matrix[id_row, id_col] = 1

    # Now if necessary add the weights and the signs

    # If it is row-stochastic, add the weights
    if row_stochastic:
        add_rs_weights2matrix(adjacency_matrix)

    # If the matrix is signed, add the negative edges
    if positive_edge_ratio < 1:
        add_signs2matrix(adjacency_matrix, positive_edge_ratio)

    return adjacency_matrix




