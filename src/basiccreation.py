import numpy as np
import random


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

    # print('f = create_random_numbers')

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
