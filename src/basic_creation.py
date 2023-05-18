"""

==============================================================
Basic Creation, (:mod:`f_abm.src.basic_creation`)
==============================================================

Description
-----------

    This module contains all the basic creation functions. It is primarily aimed at creating opinion distributions
    and agent parameters, since for digraph creation there is a separate module

Functions
---------

    - a_random_digraph
    - a_random_initial_opinion_distribution
    - a_random_inner_trait_assignation
    - create_inner_traits_local
    - create_many_opinions
    - create_many_inner_traits

"""


import random
import numpy as np
import matplotlib.pyplot as plt
from src.auxiliary_functions import (create_random_numbers, modify_opinions_method_1, modify_opinions_method_2,
                                     modify_mean, make_row_stochastic, histogram_classification)
from src.digraph_creation import small_world_digraph
from src.plot_functions import plot_inner_traits


def a_random_digraph(num_agents=10):
    """

    This function returns a random digraph, NOT a digraph with random topology, but a random digraph

    Parameters
    ----------
    num_agents: number of agents, by default 10

    Returns
    -------
    a random digraph

    """

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

    rng = np.random.default_rng()  # this is for the random numbers creation

    # Topology signature
    all_signatures = [[0, 1, 3, -5],
                      [0, 5, 10, 20],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [0, 2, 4, 6, 8, 10, 12, 14, -2, -4, -6, -8, -10, -12, -14],
                      [0, 5, 10, 15, 20, 25, -5, -10, -15, -20, -25],
                      [0, 1, 2, 3, 4, 5, -5, -10, -15, -20, -25]
                      ]

    if rng.random(1)[0] < 0.5:
        topology_sig = np.array(all_signatures[random.randint(0, len(all_signatures)-1)]) + random.randint(-10, 10)
    else:
        topology_sig = -1*np.array(all_signatures[random.randint(0, len(all_signatures) - 1)]) + random.randint(-10, 10)

    # Change probability
    if rng.random(1)[0] < 0.4:
        change_prob = float(rng.random(1)[0])
    else:
        random_param = all_param[random.randint(0, 8)]
        change_prob = create_random_numbers(num_agents=num_agents, number_parameters=random_param, limits=(0, 1))

    # Reverse probability
    if rng.random(1)[0] < 0.4:
        reverse_prob = float(rng.random(1)[0])
    else:
        random_param = all_param[random.randint(0, 8)]
        reverse_prob = create_random_numbers(num_agents=num_agents, number_parameters=random_param, limits=(0, 1))

    # Bidirectional probability
    if rng.random(1)[0] < 0.4:
        bidirectional_prob = float(rng.random(1)[0])
    else:
        random_param = all_param[random.randint(0, 8)]
        bidirectional_prob = create_random_numbers(num_agents=num_agents, number_parameters=random_param, limits=(0, 1))

    # Number random edges iterations

    digraph = small_world_digraph(num_agents=num_agents,
                                  topology_signature=topology_sig,
                                  positive_edge_ratio=0.5+(0.5*rng.random(1)[0]),
                                  change_probability=change_prob,
                                  reverse_probability=reverse_prob,
                                  bidirectional_probability=bidirectional_prob,
                                  num_random_edges_it=random.randint(0, int(np.round(0.4*num_agents*num_agents))))

    return digraph


def a_random_initial_opinion_distribution(num_agents=10):
    """

    This function returns a random initial opinion distribution

    Parameters
    ----------
    num_agents: number of agents, by default 10

    Returns
    -------
    A random initial opinion distribution

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


def a_random_inner_trait_assignation(num_agents=10):
    """

    This function returns a random inner trait assignation

    Parameters
    ----------
    num_agents: number of agents

    Returns
    -------
    Inner trait assignation

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
                 opinion_param_9]  # All parameters for the creation of the inner traits

    # Create and modify three sets of numbers which will correspond to the three types of weights
    weights_1 = create_random_numbers(num_agents=num_agents, number_parameters=all_param[random.randint(0, 8)],
                                      limits=(0, 1))
    weights_1 = modify_mean(weights_1, rng.random(1)[0], max_counter=10, epsilon=0.05, limits=(0, 1))

    weights_2 = create_random_numbers(num_agents=num_agents, number_parameters=all_param[random.randint(0, 8)],
                                      limits=(0, 1))
    weights_2 = modify_mean(weights_2, rng.random(1)[0], max_counter=10, epsilon=0.05, limits=(0, 1))

    weights_3 = create_random_numbers(num_agents=num_agents, number_parameters=all_param[random.randint(0, 8)],
                                      limits=(0, 1))
    weights_3 = modify_mean(weights_3, rng.random(1)[0], max_counter=10, epsilon=0.05, limits=(0, 1))

    random_choice = random.randint(0, 5)

    # Combine the three weights in all possible configurations and make them row-stochastic, then append only
    # the first two columns, as a reminder, the first column is the conformist weight, and the second column is
    # the radical weight. The truncation is added to make sure that the numbers are indeed between 0 and 1
    if random_choice == 0:

        inner_traits = np.concatenate((weights_1, weights_2, weights_3), axis=1)
        make_row_stochastic(inner_traits)
        inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
        return inner_traits[:, 0:2]

    elif random_choice == 1:

        inner_traits = np.concatenate((weights_1, weights_3, weights_2), axis=1)
        make_row_stochastic(inner_traits)
        inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
        return inner_traits[:, 0:2]

    elif random_choice == 2:

        inner_traits = np.concatenate((weights_2, weights_1, weights_3), axis=1)
        make_row_stochastic(inner_traits)
        inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
        return inner_traits[:, 0:2]

    elif random_choice == 3:

        inner_traits = np.concatenate((weights_2, weights_3, weights_1), axis=1)
        make_row_stochastic(inner_traits)
        inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
        return inner_traits[:, 0:2]

    elif random_choice == 4:

        inner_traits = np.concatenate((weights_3, weights_1, weights_2), axis=1)
        make_row_stochastic(inner_traits)
        inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
        return inner_traits[:, 0:2]

    elif random_choice == 5:

        inner_traits = np.concatenate((weights_3, weights_2, weights_1), axis=1)
        make_row_stochastic(inner_traits)
        inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
        return inner_traits[:, 0:2]


def create_inner_traits_local(num_agents=100):
    """

    Function to create randomly inner trait assignations, these are the agent parameters of the Classification-based
    model

    Parameters
    ----------
    num_agents: number of agents, by default 100

    Returns
    -------
    inner traits, it is a list of lists, the first element is alpha, the second is beta (the weights of the conformist
    and radical trait, respectively)

    """

    inner_traits = np.zeros((num_agents, 2))
    rng = np.random.default_rng()

    for id_agent in range(0, num_agents):
        alpha = rng.random()
        beta = rng.random()
        gamma = rng.random()

        total = alpha + beta + gamma

        alpha = alpha/total
        beta = beta / total

        inner_traits[id_agent][0] = alpha
        inner_traits[id_agent][1] = beta

    return inner_traits


def create_many_opinions(num_agents=100, file_name='standard_initial_opinions', grid=None, show_result=False):
    """

    This function creates and saves many initial opinions to be used later

    Parameters
    ----------
    num_agents: the number of agents, by default 100
    file_name: name of the file created, by default 'standard_initial_opinions'
    grid: it is the reference grid to create the initial opinions
    show_result: show the Agreement Plot of the resulting opinions. By default, it is false

    Returns
    -------

    """

    if grid is None:
        grid = np.array([[x, y] for x in np.linspace(0, 1, 41) for y in np.linspace(-1, 1, 41)  # 11 and 21
                         if (((y-x) < 0.000001) and ((y+x) > -0.000001))]).round(decimals=3)

    # opinion_param_1 = [[0, -0.5, 0.0, 1], [1, 0.9, 0.5, 2], [1, -0.5, 0.2, 10]]
    # opinion_param_2 = [[0, -0.5, 0.0, 1]]
    # opinion_param_3 = [[0, -0.5, 0.0, 1], [1, -0.9, 0.5, 10]]
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
    # https://stackoverflow.com/questions/37996295/how-to-save-numpy-array-into-computer-for-later-use-in-python

    if show_result:

        point_colors = [(0.16862745, 0.34901961, 0.76470588),
                        (0.32941176, 0.54901961, 0.18431373),
                        (0.82745098, 0.39607843, 0.50980392),
                        (0.94509804, 0.56078431, 0.00392157),
                        (0.39607843, 0.05098039, 0.10588235)]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 1, 0], [0, -1, 1, 0], linewidth=2, color=(0.2, 0.5, 0.8))
        for opinion_distribution in all_opinions:
            classification = histogram_classification(opinion_distribution)
            ax.plot(np.absolute(opinion_distribution).mean(), opinion_distribution.mean(), 'o', linewidth=1.5,
                    markersize=2, color=point_colors[classification])
        # for local_des_abs_mean, local_des_mean in grid:
        #     ax.plot(local_des_abs_mean, local_des_mean, 's',
        #             color=(0.9, 0.1, 0.3))
        ax.grid()
        plt.show()

    return all_opinions


def create_many_inner_traits(num_agents=100, file_name='standard_inner_traits', grid=None, show_result=False):
    """

    This function creates and saves many inner traits to be used later

    Parameters
    ----------
    num_agents: the number of agents
    file_name: name of the file created
    grid: it is the reference grid to create the inner traits
    show_result: a boolean determining of the resulting inner traits are shown

    Returns
    -------
    A list of numpy arrays with 'num_agents' rows and 2 columns

    """

    all_inner_traits = []
    if grid is None:
        # Create the standard grid
        grid = np.array([[x, y] for x in np.linspace(0, 1, 41) for y in np.linspace(0, 1, 41)
                         if ((y+x) < 1.000001)]).round(decimals=3)

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
                 opinion_param_9]  # All parameters for the creation of the inner traits

    for w1_des, w2_des in grid:  # For all the weights in the grid
        w3_des = 1 - (w1_des + w2_des)
        for oi_param in all_param:  # for all the parameters

            # Create and modify three sets of numbers which will correspond to the three types of weights
            weights_1 = create_random_numbers(num_agents=num_agents, number_parameters=oi_param, limits=(0, 1))
            weights_1 = modify_mean(weights_1, w1_des, max_counter=10, epsilon=0.05, limits=(0, 1))

            weights_2 = create_random_numbers(num_agents=num_agents, number_parameters=oi_param, limits=(0, 1))
            weights_2 = modify_mean(weights_2, w2_des, max_counter=10, epsilon=0.05, limits=(0, 1))

            weights_3 = create_random_numbers(num_agents=num_agents, number_parameters=oi_param, limits=(0, 1))
            weights_3 = modify_mean(weights_3, w3_des, max_counter=10, epsilon=0.05, limits=(0, 1))

            # Combine the three weights in all possible configurations and make them row-stochastic, then append only
            # the first two columns, as a reminder, the first column is the conformist weight, and the second column is
            # the radical weight. The truncation is added to make sure that the numbers are indeed between 0 and 1
            inner_traits = np.concatenate((weights_1, weights_2, weights_3), axis=1)
            make_row_stochastic(inner_traits)
            inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
            all_inner_traits.append(inner_traits[:, 0:2])

            inner_traits = np.concatenate((weights_1, weights_3, weights_2), axis=1)
            make_row_stochastic(inner_traits)
            inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
            all_inner_traits.append(inner_traits[:, 0:2])

            inner_traits = np.concatenate((weights_2, weights_1, weights_3), axis=1)
            make_row_stochastic(inner_traits)
            inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
            all_inner_traits.append(inner_traits[:, 0:2])

            inner_traits = np.concatenate((weights_2, weights_3, weights_1), axis=1)
            make_row_stochastic(inner_traits)
            inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
            all_inner_traits.append(inner_traits[:, 0:2])

            inner_traits = np.concatenate((weights_3, weights_1, weights_2), axis=1)
            make_row_stochastic(inner_traits)
            inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
            all_inner_traits.append(inner_traits[:, 0:2])

            inner_traits = np.concatenate((weights_3, weights_2, weights_1), axis=1)
            make_row_stochastic(inner_traits)
            inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
            all_inner_traits.append(inner_traits[:, 0:2])

    np.save(file_name, all_inner_traits)  # save the file as "file_name.npy"
    # to recover, use
    # all_opinions = np.load('file_name.npy')  # loads your saved array into variable all_opinions
    # https://stackoverflow.com/questions/37996295/how-to-save-numpy-array-into-computer-for-later-use-in-python

    if show_result:
        plot_inner_traits(file_name=file_name + '.npy')

    return all_inner_traits



