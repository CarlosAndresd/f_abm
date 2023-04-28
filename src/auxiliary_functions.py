"""

    Description:

        This module contains auxiliary functions.

    Functions:

        - modify_opinions_method_1
        - modify_opinions_method_2
        - add_random_edges
        - add_signs2matrix
        - add_rs_weights2matrix
        - make_row_stochastic
        - create_random_numbers


"""

import matplotlib.pyplot as plt
import numpy as np
from plot_functions import plot_histogram
import random


def modify_opinions_method_1(opinions, des_mean, des_abs_mean, epsilon=0.05, max_counter=100, show_process=False,
                             limits=(-1, 1)):
    """ This function modifies a given opinion distribution to create an opinion distribution with the desired mean
    and absolute value mean using method 1

    :param opinions: the original opinions
    :param des_mean: the desired opinion mean
    :param des_abs_mean: the desired opinion mean absolute value
    :param epsilon: the tolerance for the infinity norm
    :param max_counter: the maximum number of iterations to find the desired opinions
    :param show_process: boolean determining whether to show the creation process or not
    :param limits: a tuple with the upper and lower limits of the opinions
    :return: the new, modified opinions
    """

    if show_process:
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax2.plot([0, 1, 1, 0], [0, -1, 1, 0])
        ax2.plot(des_abs_mean, des_mean, 's', linewidth=2, color=(0.5, 0.1, 0.2))
        ax2.grid()

    opinion_mean = opinions.mean()
    opinion_abs_mean = np.absolute(opinions).mean()

    mean_difference = np.absolute(opinion_mean - des_mean)
    mean_abs_difference = np.absolute(opinion_abs_mean - des_abs_mean)

    switching_control = 0
    counter = 0
    while ((mean_difference > epsilon) or (mean_abs_difference > epsilon)) and (counter < max_counter):
        counter += 1

        if switching_control == 0:
            switching_control = 1
            # Primary mean modification (vertical)
            xi_value = np.minimum(mean_difference, 0.5)
            if opinion_mean > des_mean:
                # Reduce the opinion mean
                opinions -= xi_value
                for _ in range(5):
                    opinions[np.argmax(opinions)] = -opinions[np.argmax(opinions)]
            else:
                # Increase the opinion mean
                opinions += xi_value
                for _ in range(5):
                    opinions[np.argmin(opinions)] = -opinions[np.argmin(opinions)]
        else:
            switching_control = 0
            # Primary abs mean modification (horizontal)
            mu_value = np.minimum(mean_abs_difference, 0.3)
            if opinion_abs_mean > des_abs_mean:
                # Reduce the opinion abs mean
                opinions -= np.sign(opinions) * mu_value
                opinions *= (1-mu_value)

            else:
                # Increase the opinion abs mean
                opinions += np.sign(opinions) * mu_value
                opinions *= (1+mu_value)

        # Truncate the opinion values
        opinions = np.maximum(np.minimum(opinions, limits[1]), limits[0])

        if show_process:
            ax1.clear()
            plot_histogram(ax1, opinions)
            ax2.plot(opinion_abs_mean, opinion_mean, 'o', linewidth=2, color=(0.2, 0.5, 0.1))
            plt.gcf().canvas.draw()
            plt.pause(0.01)

        opinion_mean = opinions.mean()
        opinion_abs_mean = np.absolute(opinions).mean()

        mean_difference = np.absolute(opinion_mean - des_mean)
        mean_abs_difference = np.absolute(opinion_abs_mean - des_abs_mean)

    return opinions


def modify_opinions_method_2(opinions, des_mean, des_abs_mean, epsilon=0.05, max_counter=100, show_process=False,
                             limits=(-1, 1)):
    """ This function modifies a given opinion distribution to create an opinion distribution with the desired mean
    and absolute value mean using method 2

    :param opinions: the original opinions
    :param des_mean: the desired opinion mean
    :param des_abs_mean: the desired opinion mean absolute value
    :param epsilon: the tolerance for the infinity norm
    :param max_counter: the maximum number of iterations to find the desired opinions
    :param show_process: boolean determining whether to show the creation process or not
    :param limits: a tuple with the upper and lower limits of the opinions
    :return: the new, modified opinions
    """

    if show_process:
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax2.plot([0, 1, 1, 0], [0, -1, 1, 0])
        ax2.plot(des_abs_mean, des_mean, 's', linewidth=2, color=(0.5, 0.1, 0.2))
        ax2.grid()

    opinion_mean = opinions.mean()
    opinion_abs_mean = np.absolute(opinions).mean()

    mean_difference = np.absolute(opinion_mean - des_mean)
    mean_abs_difference = np.absolute(opinion_abs_mean - des_abs_mean)

    switching_control = 0
    counter = 0
    while ((mean_difference > epsilon) or (mean_abs_difference > epsilon)) and (counter < max_counter):
        counter += 1

        if switching_control == 0:
            switching_control = 1
            # Primary mean modification (vertical)

            m1 = (des_mean + 1) / (opinion_mean + 1)
            m2 = (des_mean - 1) / (opinion_mean - 1)
            for count, old_opinion in enumerate(opinions):
                if old_opinion <= opinion_mean:
                    opinions[count] = ((old_opinion + 1)*m1) - 1
                else:
                    opinions[count] = ((old_opinion - 1) * m2) + 1
        else:
            switching_control = 0
            # Primary abs mean modification (horizontal)
            mu_value = np.minimum(mean_abs_difference, 0.3)
            if opinion_abs_mean > des_abs_mean:
                # Reduce the opinion abs mean by contracting the opinions
                # remove the mean
                local_opinion_mean = opinions.mean()
                opinions -= local_opinion_mean
                # contract the opinions
                opinions *= (1-mu_value)
                # restore the mean
                opinions += local_opinion_mean

            else:
                # Increase the opinion abs mean by expanding the opinions
                # remove the mean
                local_opinion_mean = opinions.mean()
                opinions -= local_opinion_mean
                # expand the opinions
                opinions *= (1 + mu_value)
                # restore the mean
                opinions += local_opinion_mean

        # Truncate the opinion values
        opinions = np.maximum(np.minimum(opinions, limits[1]), limits[0])

        if show_process:
            ax1.clear()
            plot_histogram(ax1, opinions)
            ax2.plot(opinion_abs_mean, opinion_mean, 'o', linewidth=2, color=(0.2, 0.5, 0.1))
            plt.gcf().canvas.draw()
            plt.pause(0.01)

        opinion_mean = opinions.mean()
        opinion_abs_mean = np.absolute(opinions).mean()

        mean_difference = np.absolute(opinion_mean - des_mean)
        mean_abs_difference = np.absolute(opinion_abs_mean - des_abs_mean)

    return opinions


def add_random_edges(adjacency_matrix, num_iterations=10, default_type=0):
    """
    Function to add random edges to the adjacency matrix 'adjacency_matrix', the edges have no weight or sign.
    The function does not guarantee that these are new edges, it randomly selects cells of the adjacency matrix and
    adds edges

    :param adjacency_matrix: the adjacency matrix to be modified
    :param num_iterations: the number of iterations
    :param default_type: the ID of the default digraph
    :return:
    """

    # Get the number of agents
    num_agents = adjacency_matrix.shape[0]

    for _ in range(0,num_iterations):
        randon_row = random.randint(0, num_agents-1)
        randon_col = random.randint(0, num_agents-1)

        adjacency_matrix[randon_row][randon_col] = 1.0


def add_signs2matrix(adjacency_matrix, positive_edge_ratio):
    """ Function that adds the signs to the adjacency matrix of a signed digraph

    :param adjacency_matrix: current adjacency matrix, presumably with only non-negative weights
    :param positive_edge_ratio: ratio of positive edges
    :return: There is no need to return anything, since the function modifies the adjacency matrix as desired
    """

    # print('f = add_signs2matrix')

    # Get the number of agents
    num_agents = adjacency_matrix.shape[0]

    # Total number of edges (excluding self-loops)
    num_edges = (adjacency_matrix != 0).sum() - num_agents

    # Approximate the number of negative edges
    neg_edges = int(np.floor(positive_edge_ratio * num_edges))

    # List all the non self-loop edges
    edges = [[id_row, id_col] for id_row in range(num_agents) for id_col in range(num_agents)
             if (id_row != id_col and adjacency_matrix[id_row, id_col] != 0)]

    # Sort them randomly
    rng = np.random.default_rng()
    rng.shuffle(edges)
    edges = np.array(edges)[:neg_edges, :]

    # Change the sign of the edge
    for id_row, id_col in edges:
        adjacency_matrix[id_row, id_col] *= -1


def add_rs_weights2matrix(adjacency_matrix):
    """ Function that adds the stochastic weights to an adjacency matrix

    :param adjacency_matrix: original adjacency matrix
    :return: there is no need to return anything, as the adjacency matrix is transformed in the function
    """

    # print('f = add_rs_weights2matrix')

    # Get the number of agents
    num_agents = adjacency_matrix.shape[0]

    # Multiply the adjacency matrix by random numbers

    for id_row in range(0, adjacency_matrix.shape[0]):
        adjacency_matrix[id_row, :] = adjacency_matrix[id_row, :] * np.random.uniform(low=0.0, high=1.0, size=(1, num_agents))

    # Make the matrix row-stochastic
    make_row_stochastic(adjacency_matrix)


def make_row_stochastic(matrix):

    # print('f = make_row_stochastic')

    # Function that takes a matrix and makes it row-stochastic
    for id_row in range(0, matrix.shape[0]):
        denominator = matrix[id_row, :].sum()
        if denominator == 0:
            # If the denominator is zero, then amke every element in the row have the same weight
            matrix[id_row, :] = np.ones(matrix.shape[1])*(1/matrix.shape[1])
        else:
            matrix[id_row, :] = matrix[id_row, :] * (1 / denominator)


def create_random_numbers(num_agents=100, number_parameters=None, limits=(-1, 1)):
    """ This function creates and returns a list of random 'num_agents' numbers. This function is used to create initial
        opinions and also to create agent parameters. Its default use is to create initial opinions

    :param num_agents: number of opinions that are returned, by default 100
    :param number_parameters: a lists of lists, every element corresponds to a different set of initial opinions to be
        created. Each element of this list contains 4 elements:
            [0] -> signals the type of distribution to create.  '0' is a uniform distribution
                                                                'any non 0' is a normal distribution
            [1] -> if the distribution is uniform, this is the minimum value. If the distribution is normal, this is the
                    mean
            [2] -> if the distribution is uniform, this is the maximum value. If the distribution is normal, this is the
                    variance
            [3] -> the fraction of all the agents. The sum does not have to add to one, as it will be normalized

    :param limits: a tuple of two numbers, the first is the lower bound and the second the upper bound
    :return: numpy array of 'num_agents' rows and 1 column (a list of lists) of opinions
    """

    rng = np.random.default_rng()  # this is for the random numbers creation

    if number_parameters is None:
        number_parameters = [[0, -1.0, 1.0, 1]]

    # for ease, transform the list of lists to a numpy 2d array
    number_parameters = np.array(number_parameters)

    # the first thing to do is to compute the number of agents each sub-distribution will have, start by normalizing the
    # fractions
    fractions = number_parameters[:, 3]
    fractions = fractions/fractions.sum()
    sub_num_agents = np.floor(fractions*num_agents)  # compute the number of agents this assignation corresponds to
    missing_agents = num_agents-sub_num_agents.sum()  # number of agents that are left to assign
    while missing_agents > 0:
        # randomly assign one agent to one subgroup
        random_index = random.randint(0, (len(sub_num_agents)-1))
        sub_num_agents[random_index] += 1
        missing_agents = num_agents - sub_num_agents.sum()  # number of agents that are left to assign

    # replace the fraction, for the actual number of agents
    number_parameters[:, 3] = sub_num_agents

    initial_opinions = None
    for subgroup_info in number_parameters:
        if subgroup_info[0] == 0:
            # create a uniform distribution
            min_op = subgroup_info[1]
            max_op = subgroup_info[2]
            local_opinions = (max_op - min_op) * rng.random((int(subgroup_info[3]), 1)) + min_op

        else:
            # create a normal distribution
            dist_mean = subgroup_info[1]
            dist_variance = subgroup_info[2]
            local_opinions = rng.normal(dist_mean, dist_variance, (int(subgroup_info[3]), 1))

        if initial_opinions is None:
            initial_opinions = local_opinions
        else:
            initial_opinions = np.concatenate((initial_opinions, local_opinions))

    # Truncate the opinion values
    initial_opinions = np.maximum(np.minimum(initial_opinions, limits[1]), limits[0])

    return initial_opinions

