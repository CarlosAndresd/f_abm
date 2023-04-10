import random
from auxiliary_functions import *


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
