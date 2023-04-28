"""

    Description:

        This module contains all the basic creation functions. It is primarily aimed at creating opinion distributions
        and agent parameters, since for digraph creation there is a separate module

    Functions:

        - Related to initial opinions:
            - a_random_initial_opinion_distribution
            - create_many_opinions

        - Related to digraphs:
            - default_digraph
            - ring_digraph
            - random_digraph
            - small_world_digraph

        - Related to agent parameters:







"""

import random
from auxiliary_functions import *


def a_random_initial_opinion_distribution(num_agents=10):
    """
    This function returns a random initial opinion distribution
    :param num_agents: number of agents
    :return: a random initial opinion distribution
    """

    rng = np.random.default_rng()  # this is for the random numbers creation

    opinion_param_1 = [[0, -1.0, 1.0, 1]]
    opinion_param_2 = [[0, 0.0, 1.0, 1]]
    opinion_param_3 = [[0, -1.0, 0.0, 1]]
    opinion_param_4 = [[1, 0.0, 1.0, 1]]
    opinion_param_5 = [[1, -0.5, 0.5, 1], [1, 0.5, 0.5, 1]]
    opinion_param_6 = [[0, -1.0, -0.5, 1], [0, 0.5, 1.0, 1]]
    opinion_param_7 = [[0, -1.0, -0.7, 1], [1, 0.5, 0.5, 1]]
    opinion_param_8 = [[1, -0.5, 0.5, 1], [0, 0.7, 1.0, 1]]
    opinion_param_9 = [[0, -1.0, -0.7, 1], [0, -0.2, 0.2, 1], [0, 0.7, 1.0, 1]]
    all_param = [opinion_param_1,
                 opinion_param_2,
                 opinion_param_3,
                 opinion_param_4,
                 opinion_param_5,
                 opinion_param_6,
                 opinion_param_7,
                 opinion_param_8,
                 opinion_param_9]

    local_des_abs_mean = rng.random(1)[0]
    local_des_mean = rng.random(1)[0]*local_des_abs_mean

    initial_opinions = create_random_numbers(num_agents=num_agents, number_parameters=all_param[random.randint(0, 8)])

    if rng.random(1)[0] > 0.5:
        return modify_opinions_method_1(initial_opinions, des_mean=local_des_mean,
                                        des_abs_mean=local_des_abs_mean, epsilon=0.02)

    else:
        return modify_opinions_method_2(initial_opinions, des_mean=local_des_mean,
                                        des_abs_mean=local_des_abs_mean, epsilon=0.02)


def create_many_opinions(num_agents=100, file_name='standard_initial_opinions', grid=None, show_result=False):
    """ This function creates and saves many initial opinions to be used later

    :param num_agents: the number of agents
    :param file_name: name of the file created
    :param grid: it is the reference grid to create the initial opinions
    :param show_result: show the Agreement Plot of the resulting opinions. By default, it is false
    :return:
    """

    if grid is None:
        grid = np.array([[x, y] for x in np.linspace(0, 1, 41) for y in np.linspace(-1, 1, 41)  # 11 and 21
                         if (((y-x) < 0.000001) and ((y+x) > -0.000001))]).round(decimals=3)

    opinion_param_1 = [[0, -1.0, 1.0, 1]]
    opinion_param_2 = [[0, 0.0, 1.0, 1]]
    opinion_param_3 = [[0, -1.0, 0.0, 1]]
    opinion_param_4 = [[1, 0.0, 1.0, 1]]
    opinion_param_5 = [[1, -0.5, 0.5, 1], [1, 0.5, 0.5, 1]]
    opinion_param_6 = [[0, -1.0, -0.5, 1], [0, 0.5, 1.0, 1]]
    opinion_param_7 = [[0, -1.0, -0.7, 1], [1, 0.5, 0.5, 1]]
    opinion_param_8 = [[1, -0.5, 0.5, 1], [0, 0.7, 1.0, 1]]
    opinion_param_9 = [[0, -1.0, -0.7, 1], [0, -0.2, 0.2, 1], [0, 0.7, 1.0, 1]]
    all_param = [opinion_param_1,
                 opinion_param_2,
                 opinion_param_3,
                 opinion_param_4,
                 opinion_param_5,
                 opinion_param_6,
                 opinion_param_7,
                 opinion_param_8,
                 opinion_param_9]

    all_opinions = []
    for local_des_abs_mean, local_des_mean in grid:
        for oi_param in all_param:
            initial_opinions = create_random_numbers(num_agents=num_agents, number_parameters=oi_param)
            new_opinions = modify_opinions_method_1(initial_opinions, des_mean=local_des_mean,
                                                    des_abs_mean=local_des_abs_mean, epsilon=0.02)
            all_opinions.append(new_opinions)

            new_opinions = modify_opinions_method_2(initial_opinions, des_mean=local_des_mean,
                                                    des_abs_mean=local_des_abs_mean, epsilon=0.02)
            all_opinions.append(new_opinions)

    np.save(file_name, all_opinions)  # save the file as "file_name.npy"
    # to recover, use
    # all_opinions = np.load('file_name.npy')  # loads your saved array into variable all_opinions

    if show_result:

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 1, 0], [0, -1, 1, 0], linewidth=2, color=(0.2, 0.5, 0.8))
        for opinion_distribution in all_opinions:
            ax.plot(np.absolute(opinion_distribution).mean(), opinion_distribution.mean(), 'o', linewidth=1.5,
                    markersize=2)
        ax.grid()
        plt.show()

    return all_opinions


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
            if num_agents <= absolute_neighbour:
                absolute_neighbour = absolute_neighbour - num_agents
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


def default_digraph(default_type=0, num_agents=10):
    """
    This function returns pre-made digraphs to be used primarily as default for functions. The pre-made digraph that
    will be called for default will always be the one with default_type=0
    :param default_type: ID of the default digraph
    :param num_agents: number of agents
    :return: the corresponding adjacency matrix
    """

    # print('f = default_digraph')

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
