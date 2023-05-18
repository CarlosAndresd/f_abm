"""

==============================================================
Data Analysis, (:mod:`f_abm.src.data_analysis_functions`)
==============================================================

Description
-----------

    This module contains all the data analysis related functions

Functions
---------

    - gather_data
    - obtain_features
    - feature_computation
    - compute_mean_opinion_difference
    - compute_opinion_metrics_by_agent_type
    - compute_trait_allocation_metrics
    - compute_opinion_metrics
    - compute_inner_trait_metrics
    - compute_digraph_metrics
    - compute_balance_index
    - compute_bidirectional_coefficient
    - compute_degrees
    - compute_clustering

"""


import random
import numpy as np
import pandas as pd
from src.basiccreation import (create_many_inner_traits, create_many_opinions, a_random_digraph,
                                a_random_initial_opinion_distribution, a_random_inner_trait_assignation, )
from src.model_functions import model_evolution
from src.auxiliary_functions import (histogram_classification, matrix_exp, digraph2topology, )
from src.digraph_creation import (default_digraph, )


def gather_data(num_agents=1000, num_iterations=1000, global_name='default_name'):
    _ = create_many_inner_traits(num_agents=num_agents, file_name=global_name+'_traits')
    _ = create_many_opinions(num_agents=num_agents, file_name=global_name+'_opinions')
    obtain_features(num_agents=num_agents, num_iterations=num_iterations, file_name=global_name+'_data',
                    traits_file_name=global_name+'_traits', opinions_file_name=global_name+'_opinions')


def obtain_features(num_agents=1000, num_iterations=100, file_name=None, traits_file_name=None,
                    opinions_file_name=None):
    """
    File to create the data for the training, validation, and testing

    :param num_agents: number of agents in all the simulations
    :param num_iterations: number of iterations
    :param file_name: name of the Excel data to print the output to
    :return:
    """
    # column_names = ['balance index',  # balance index,
    #                 'bidirectional coefficient',  # bidirectional coefficient,
    #                 'mean in-degree',  # mean in-degree,
    #                 'in-degree variance',  # in-degree variance,
    #                 'mean out-degree',  # mean out-degree,
    #                 'out-degree variance',  # out-degree variance,
    #                 'mean cluster',  # mean cluster,
    #                 'variance cluster',  # variance cluster
    #                 'mean initial opinions',  # mean initial opinions
    #                 'mean abs initial opinions',  # mean abs initial opinions
    #                 'number conformist agents',  # num_con_agents,
    #                 'number radical agents',  # num_rad_agents,
    #                 'number stubborn agents',  # num_stb_agents,
    #                 'average conformist weight',  # av_con,
    #                 'average radical weight',  # av_rad,
    #                 'average stubborn weight',  # av_stb
    #                 'mean opinion digraph difference',  # mean_opinion_difference
    #                 'number conformist agents',  # num_con_agents,
    #                 'mean initial opinion conformist agents',  # opinion_mean_con_agents,
    #                 'mean abs initial opinion conformist agents',  # abs_opinion_mean_con_agents,
    #                 'number radical agents',  # num_rad_agents,
    #                 'mean initial opinion radical agents',  # opinion_mean_rad_agents,
    #                 'mean abs initial opinion radical agents',  # abs_opinion_mean_rad_agents,
    #                 'number stubborn agents',  # num_stb_agents,
    #                 'mean initial opinion stubborn agents',  # opinion_mean_stb_agents,
    #                 'mean abs initial opinion stubborn agents',  # abs_opinion_mean_stb_agents,
    #                 'mean difference trait allocation',  # mean_difference
    #                 'type final opinions',  # type
    #                 'mean final opinions',  # mean final opinions
    #                 'mean abs final opinions']  # mean abs final opinions

    column_names = ['bal_ind',  # balance index,
                    'bid_coe',  # bidirectional coefficient,
                    'mean_in_d',  # mean in-degree,
                    'var_in_d',  # in-degree variance,
                    'mean_out_d',  # mean out-degree,
                    'var_out_d',  # out-degree variance,
                    'mean_clu',  # mean cluster,
                    'var_clu',  # variance cluster
                    'mean_ini_op',  # mean initial opinions
                    'mean_abs_ini_op',  # mean abs initial opinions
                    'num_con',  # num_con_agents,
                    'num_rad',  # num_rad_agents,
                    'num_stb',  # num_stb_agents,
                    'av_con',  # av_con,
                    'av_rad',  # av_rad,
                    'av_stb',  # av_stb
                    'mean_op_di_diff',  # mean_opinion_difference
                    'num_con_n',  # num_con_agents,
                    'mean_op_con',  # opinion_mean_con_agents,
                    'mean_abs_op_con',  # abs_opinion_mean_con_agents,
                    'num_rad_n',  # num_rad_agents,
                    'mean_op_rad',  # opinion_mean_rad_agents,
                    'mean_abs_op_rad',  # abs_opinion_mean_rad_agents,
                    'num_stb_n',  # num_stb_agents,
                    'mean_op_stb',  # opinion_mean_stb_agents,
                    'mean_abs_op_stb',  # abs_opinion_mean_stb_agents,
                    'mean_diff_trait',  # mean_difference
                    'type_final',  # type
                    'mean_fin_op',  # mean final opinions
                    'mean_abs_fin_op']  # mean abs final opinions

    if traits_file_name is None:
        traits_file_name = 'default_name_traits'

    if opinions_file_name is None:
        opinions_file_name = 'default_name_opinions'

    # Load the set of possible initial opinions (100 agents)
    all_opinions = np.load(opinions_file_name+'.npy')
    num_opinions = np.shape(all_opinions)[0]  # Number of possible initial opinions

    # Load the set of possible inner traits (100 agents)
    all_inner_traits = np.load(traits_file_name+'.npy')
    num_inner_traits = np.shape(all_inner_traits)[0]  # Number of possible inner traits

    results = np.expand_dims(
        feature_computation(num_agents=num_agents, adjacency_matrix=None,
                            opinion_distribution=all_opinions[random.randint(0, num_opinions-1)],
                            inner_trait_assignations=all_inner_traits[random.randint(0, num_inner_traits-1)]), axis=0)

    for id_row in range(1, num_iterations):
        print(f'Current row = {id_row}')
        new_results = np.expand_dims(
            feature_computation(num_agents=num_agents, adjacency_matrix=None,
                                opinion_distribution=all_opinions[random.randint(0, num_opinions - 1)],
                                inner_trait_assignations=all_inner_traits[random.randint(0, num_inner_traits - 1)]),
            axis=0)
        results = np.concatenate((results, new_results), axis=0)

    df = pd.DataFrame(results, columns=column_names)

    if file_name is None:
        file_name = 'output'

    df.to_excel(file_name + '.xlsx')


def feature_computation(num_agents=10, print_information=False, adjacency_matrix=None, opinion_distribution=None,
                        inner_trait_assignations=None):

    # print('f = feature_computation')

    if adjacency_matrix is None:
        adjacency_matrix = a_random_digraph(num_agents=num_agents)

    if opinion_distribution is None:
        opinion_distribution = a_random_initial_opinion_distribution(num_agents=num_agents)

    if inner_trait_assignations is None:
        inner_trait_assignations = a_random_inner_trait_assignation(num_agents=num_agents)

    # Shuffle the opinion distribution and inner_trait_assignations
    rng = np.random.default_rng()
    rng.shuffle(opinion_distribution)
    rng.shuffle(inner_trait_assignations)

    final_opinions = model_evolution(initial_opinions=opinion_distribution,
                                     adjacency_matrix=adjacency_matrix,
                                     agent_parameters=inner_trait_assignations,
                                     num_steps=50)

    # Digraph metrics
    digraph_metrics = compute_digraph_metrics(adjacency_matrix=adjacency_matrix,
                                              print_information=print_information)

    # balance index, bidirectional coefficient, mean in-degree, in-degree variance, mean
    # out-degree, out-degree variance, mean cluster, variance cluster

    # Opinion metrics
    initial_opinion_metrics = compute_opinion_metrics(opinion_distribution=opinion_distribution,
                                                      print_information=print_information)

    # mean initial opinions
    # mean abs initial opinions

    final_opinion_metrics = compute_opinion_metrics(opinion_distribution=final_opinions,
                                                    print_information=print_information)

    # mean final opinions
    # mean abs final opinions

    # Inner trait assignation metrics
    inner_trait_metrics = compute_inner_trait_metrics(inner_traits=inner_trait_assignations,
                                                      print_information=print_information)

    # num_con_agents, num_rad_agents, num_stb_agents, av_con, av_rad, av_stb

    # Digraph and opinion metrics
    mean_opinion_difference = compute_mean_opinion_difference(adjacency_matrix=adjacency_matrix,
                                                              opinion_distribution=opinion_distribution,
                                                              print_information=print_information)

    # mean_opinion_difference

    # Opinion and traits metrics
    opinion_metrics_by_agent_type = compute_opinion_metrics_by_agent_type(opinion_distribution=opinion_distribution,
                                                                          inner_traits=inner_trait_assignations,
                                                                          print_information=print_information)

    # num_con_agents, opinion_mean_con_agents, abs_opinion_mean_con_agents, num_rad_agents,
    # opinion_mean_rad_agents, abs_opinion_mean_rad_agents, num_stb_agents, opinion_mean_stb_agents,
    # abs_opinion_mean_stb_agents,

    # Traits and digraph metrics
    trait_allocation_metrics = compute_trait_allocation_metrics(adjacency_matrix=adjacency_matrix,
                                                                inner_traits=inner_trait_assignations,
                                                                print_information=print_information)

    # mean_difference

    final_type = np.array([histogram_classification(final_opinions)])

    all_features = np.concatenate((digraph_metrics,
                                   initial_opinion_metrics,
                                   inner_trait_metrics,
                                   mean_opinion_difference,
                                   opinion_metrics_by_agent_type,
                                   trait_allocation_metrics,
                                   final_type,
                                   final_opinion_metrics,
                                   ))

    return all_features


def compute_mean_opinion_difference(adjacency_matrix=None, opinion_distribution=None, num_agents=10,
                                    print_information=False):
    """
    This function computes the mean opinion difference, given an adjacency matrix and an opinion distribution
    :param adjacency_matrix: adjacency matrix
    :param opinion_distribution: opinion distribution
    :param num_agents: number of agents
    :param print_information: boolean determining whether the metric is shown or not
    :return: mean opinion difference
    """

    # print('f = compute_mean_opinion_difference')

    if adjacency_matrix is None:
        adjacency_matrix = a_random_digraph(num_agents=num_agents)
    else:
        # Get the number of agents
        num_agents = adjacency_matrix.shape[0]

    if opinion_distribution is None:
        opinion_distribution = a_random_initial_opinion_distribution(num_agents=num_agents)

    if num_agents != opinion_distribution.shape[0]:
        # The adjacency matrix and opinion distribution have incompatible dimensions
        print('The adjacency matrix and opinion distribution have incompatible dimensions')
        return None

    # List all the edges, excluding self loops
    edges = [[id_row, id_col] for id_row in range(num_agents) for id_col in range(num_agents)
             if (id_row != id_col and adjacency_matrix[id_row, id_col] != 0)]

    num_edges = 0
    total_difference = 0
    for id_row, id_col in edges:
        num_edges += 1
        total_difference += np.abs(opinion_distribution[id_row] - opinion_distribution[id_col])

    if num_edges > 0:
        mean_opinion_difference = total_difference/num_edges
    else:
        # This would only happen if the digraph is completely disconnected
        mean_opinion_difference = 0

    if print_information:
        print(f'mean opinion digraph difference {mean_opinion_difference}')

    return mean_opinion_difference


def compute_opinion_metrics_by_agent_type(opinion_distribution=None, inner_traits=None, num_agents=10,
                                          print_information=False):

    """ This function computes the opinion metric by agent type

    :param opinion_distribution: the opinion distribution
    :param inner_traits: the inner traits
    :param num_agents: the number of agents
    :param print_information: boolean determining whether the metric is shown or not
    :return: the opinion metric by agent type
    """

    # print('f = compute_opinion_metrics_by_agent_type')

    # first, classify each agent, depending on which inner trait has the greatest weight

    if opinion_distribution is None:
        opinion_distribution = a_random_initial_opinion_distribution(num_agents=num_agents)
    else:
        # Get the number of agents
        num_agents = opinion_distribution.shape[0]

    if inner_traits is None:
        inner_traits = a_random_inner_trait_assignation(num_agents=num_agents)

    if num_agents != inner_traits.shape[0]:
        # The adjacency matrix and opinion distribution have incompatible dimensions
        print('The inner traits and opinion distribution have incompatible dimensions')
        return None

    con_agents = []
    rad_agents = []
    stb_agents = []

    for id_agent in range(0, num_agents):
        con_trait = inner_traits[id_agent][0]
        rad_trait = inner_traits[id_agent][1]
        stb_trait = 1 - (con_trait + rad_trait)

        if (con_trait > rad_trait) and (con_trait > stb_trait):
            con_agents.append(id_agent)
        elif (rad_trait > con_trait) and (rad_trait > stb_trait):
            rad_agents.append(id_agent)
        elif (stb_trait > con_trait) and (rad_trait > stb_trait):
            stb_agents.append(id_agent)
        else:
            stb_agents.append(id_agent)

    num_con_agents = len(con_agents)
    if num_con_agents > 0:
        opinion_mean_con_agents = opinion_distribution[con_agents].mean()
        abs_opinion_mean_con_agents = np.abs(opinion_distribution[con_agents]).mean()
    else:
        opinion_mean_con_agents = 0
        abs_opinion_mean_con_agents = 0

    num_rad_agents = len(rad_agents)
    if num_rad_agents > 0:
        opinion_mean_rad_agents = opinion_distribution[rad_agents].mean()
        abs_opinion_mean_rad_agents = np.abs(opinion_distribution[rad_agents]).mean()
    else:
        opinion_mean_rad_agents = 0
        abs_opinion_mean_rad_agents = 0

    num_stb_agents = len(stb_agents)
    if num_stb_agents > 0:
        opinion_mean_stb_agents = opinion_distribution[stb_agents].mean()
        abs_opinion_mean_stb_agents = np.abs(opinion_distribution[stb_agents]).mean()
    else:
        opinion_mean_stb_agents = 0
        abs_opinion_mean_stb_agents = 0

    if print_information:
        print(f'number conformist agents {num_con_agents}')
        print(f'opinion mean conformist agents {opinion_mean_con_agents}')
        print(f'absolute value opinion mean conformist agents {abs_opinion_mean_con_agents}')

        print(f'number radical agents {num_rad_agents}')
        print(f'opinion mean radical agents {opinion_mean_rad_agents}')
        print(f'absolute value opinion mean radical agents {abs_opinion_mean_rad_agents}')

        print(f'number stubborn agents {num_stb_agents}')
        print(f'opinion mean stubborn agents {opinion_mean_stb_agents}')
        print(f'absolute value opinion mean stubborn agents {abs_opinion_mean_stb_agents}')

    metrics = [num_con_agents,
               opinion_mean_con_agents,
               abs_opinion_mean_con_agents,
               num_rad_agents,
               opinion_mean_rad_agents,
               abs_opinion_mean_rad_agents,
               num_stb_agents,
               opinion_mean_stb_agents,
               abs_opinion_mean_stb_agents,
               ]

    return np.array(metrics)


def compute_trait_allocation_metrics(adjacency_matrix=None, inner_traits=None, num_agents=10, print_information=False):
    """
    This function computes the mean inner trait difference between neighbours in the digraph

    :param adjacency_matrix: the corresponding digraph
    :param inner_traits: the corresponding inner trait assignation
    :param num_agents: the number of agents
    :param print_information: boolean determining whether the metric is shown or not
    :return: the mean inner trait difference between neighbours in the digraph
    """

    # print('f = compute_trait_allocation_metrics')

    if inner_traits is None:
        inner_traits = a_random_inner_trait_assignation(num_agents=num_agents)
    else:
        # Get the number of agents
        num_agents = inner_traits.shape[0]

    if adjacency_matrix is None:
        adjacency_matrix = a_random_digraph(num_agents=num_agents)

    if num_agents != adjacency_matrix.shape[0]:
        # The inner trait assignation and adjacency matrix have incompatible dimensions
        print('The inner trait assignation and adjacency matrix have incompatible dimensions')
        return None

    # List all the edges, excluding self loops
    edges = [[id_row, id_col] for id_row in range(num_agents) for id_col in range(num_agents)
             if (id_row != id_col and adjacency_matrix[id_row, id_col] != 0)]

    num_edges = 0
    total_difference = 0
    for id_row, id_col in edges:
        num_edges += 1

        con_src = inner_traits[id_row][0]
        con_trg = inner_traits[id_col][0]

        rad_src = inner_traits[id_row][1]
        rad_trg = inner_traits[id_col][1]

        stb_src = 1 - (con_src + rad_src)
        stb_trg = 1 - (con_trg + rad_trg)

        con_diff = con_src - con_trg
        rad_diff = rad_src - rad_trg
        stb_diff = stb_src - stb_trg

        total_difference += np.sqrt((con_diff*con_diff)+(rad_diff*rad_diff)+(stb_diff*stb_diff))

    if num_edges > 0:
        mean_difference = total_difference / num_edges
    else:
        # This would only happen if the digraph is completely disconnected
        mean_difference = 0

    if print_information:
        print(f'mean inner trait assignation difference {mean_difference}')

    return np.array([mean_difference])


def compute_opinion_metrics(opinion_distribution=None, num_agents=10, print_information=False):
    """
    This function computes the mean and mean of the absolute value of the opinion distribution

    :param opinion_distribution: the opinion distribution
    :param num_agents: number of agents
    :param print_information: boolean determining whether the print the information or not
    :return: the mean and mean of the absolute value of the opinion distribution
    """

    # print('f = compute_opinion_metrics')

    if opinion_distribution is None:
        opinion_distribution = a_random_initial_opinion_distribution(num_agents=num_agents)

    mean_opinions = opinion_distribution.mean()
    mean_abs_opinions = np.abs(opinion_distribution).mean()

    if print_information:
        print(f'opinion mean {mean_opinions}')
        print(f'absolute value opinion mean {mean_abs_opinions}')

    return np.array([mean_opinions, mean_abs_opinions])


def compute_inner_trait_metrics(inner_traits=None, num_agents=10, print_information=False):
    """
    This function computes the metrics of the inner trait assignation

    :param inner_traits: the inner trait assignation
    :param num_agents: the number of agents
    :param print_information: boolean that determines if the information is printed
    :return:
    """

    # print('f = compute_inner_trait_metrics')

    if inner_traits is None:
        inner_traits = a_random_inner_trait_assignation(num_agents=num_agents)
    else:
        # Get the number of agents
        num_agents = inner_traits.shape[0]

    av_con, av_rad = np.maximum(np.minimum(inner_traits.mean(axis=0), 1), 0)
    av_stb = 1 - (av_con + av_rad)

    con_agents = []
    rad_agents = []
    stb_agents = []

    for id_agent in range(0, num_agents):
        con_trait = inner_traits[id_agent][0]
        rad_trait = inner_traits[id_agent][1]
        stb_trait = 1 - (con_trait + rad_trait)

        if (con_trait > rad_trait) and (con_trait > stb_trait):
            con_agents.append(id_agent)
        elif (rad_trait > con_trait) and (rad_trait > stb_trait):
            rad_agents.append(id_agent)
        elif (stb_trait > con_trait) and (rad_trait > stb_trait):
            stb_agents.append(id_agent)
        else:
            stb_agents.append(id_agent)

    num_con_agents = len(con_agents)
    num_rad_agents = len(rad_agents)
    num_stb_agents = len(stb_agents)

    if print_information:
        print(f'number conformist agents {num_con_agents}')
        print(f'number radical agents {num_rad_agents}')
        print(f'number stubborn agents {num_stb_agents}')

        print(f'average conformist trait {av_con}')
        print(f'average radical trait {av_rad}')
        print(f'average stubborn trait {av_stb}')

    return np.array([num_con_agents, num_rad_agents, num_stb_agents, av_con, av_rad, av_stb])


def compute_digraph_metrics(adjacency_matrix=None, default_type=0, print_information=False):
    """
    This is a function used to compute several digraph metrics at once

    :param adjacency_matrix: the adjacency matrix for which the metrics will be computed
    :param default_type: ID of the default adjacency matrix
    :param print_information: whether to print information or not
    :return:
    """

    # print('f = compute_digraph_metrics')

    if adjacency_matrix is None:
        adjacency_matrix = default_digraph(default_type=default_type)

    degree_metrics = compute_degrees(adjacency_matrix=adjacency_matrix,
                                     print_information=print_information)  # Degree metrics
    cluster_metrics = compute_clustering(adjacency_matrix=adjacency_matrix,
                                         print_information=print_information) # Clustering metrics

    metrics = [compute_balance_index(adjacency_matrix=adjacency_matrix,
                                     print_information=print_information),  # Balance index
               compute_bidirectional_coefficient(adjacency_matrix=adjacency_matrix,
                                                 print_information=print_information),  # Bidirectional coefficient
               degree_metrics[0], degree_metrics[1], degree_metrics[2], degree_metrics[3],
               cluster_metrics[0], cluster_metrics[1],
               ]

    return np.array(metrics)  # balance index, bidirectional coefficient, mean in-degree, in-degree variance, mean
                                # out-degree, out-degree variance, mean cluster, variance cluster


def compute_balance_index(adjacency_matrix=None, default_type=0, print_information=False):
    """
    Function to approximate the balance index of a signed network
    :param adjacency_matrix: the adjacency matrix
    :param default_type: ID of the default digraph
    :param print_information: Boolean determining if the computed values are printed
    :return: the balance index
    """

    # print('f = compute_balance_index')

    if adjacency_matrix is None:
        adjacency_matrix = default_digraph(default_type=default_type)

    balance_index = (matrix_exp(adjacency_matrix).trace())/(matrix_exp(np.absolute(adjacency_matrix)).trace())

    if print_information:
        print(f'The Balance index is {balance_index}')

    return balance_index


def compute_bidirectional_coefficient(adjacency_matrix=None, default_type=0, print_information=False):
    """
    This function computes the bidirectional coefficient of a given adjacency matrix
    :param adjacency_matrix: the adjacency matrix.
    :param default_type: ID of the default digraph
    :param print_information: Boolean determining if the computed values are printed
    :return: a float between 0.0 and 1.0 with the bidirectional coefficient
    """

    # print('f = compute_bidirectional_coefficient')

    if adjacency_matrix is None:
        adjacency_matrix = default_digraph(default_type=default_type)

    # Get the number of agents
    num_agents = adjacency_matrix.shape[0]

    num_edges = 0
    num_bidirectional_edges = 0

    for id_row in range(0, num_agents):
        for id_col in range(0, num_agents):
            if adjacency_matrix[id_row][id_col] != 0.0:
                if id_row != id_col:
                    num_edges += 1
                    if adjacency_matrix[id_col][id_row] != 0.0:
                        num_bidirectional_edges += 1

    if num_edges == 0.0:
        bidirectional_coefficient = 0
    else:
        bidirectional_coefficient = num_bidirectional_edges/num_edges

    if print_information:
        print(f'The Bidirectional coefficient is {bidirectional_coefficient}')

    return bidirectional_coefficient


def compute_degrees(adjacency_matrix=None, default_type=0, print_information=False):
    """
    Function used to compute the metrics related to the degree of the nodes, namely, the mean and variance of the in and
    out degrees
    :param adjacency_matrix: the adjacency matrix
    :param default_type: ID of the default digraph
    :param print_information: Boolean determining if the computed values are printed
    :return: a numpy array with 4 numbers corresponding, in order, to the mean in-degree, in-degree variance, mean
    out-degree, and out-degree variance
    """

    # print('f = compute_degrees')

    if adjacency_matrix is None:
        adjacency_matrix = default_digraph(default_type=default_type)

    topology = digraph2topology(adjacency_matrix=adjacency_matrix)
    transpose_topology = topology.transpose()

    in_degree = np.array([(in_neigh.sum()-1) for in_neigh in topology])
    out_degree = np.array([(out_neigh.sum()-1) for out_neigh in transpose_topology])
    # 1 is subtracted to account for the self-loop

    mean_in_degree = in_degree.mean()
    var_in_degree = in_degree.var()

    mean_out_degree = out_degree.mean()
    var_out_degree = out_degree.var()

    if print_information:
        print(f'The mean in-degree is {mean_in_degree}')
        print(f'The in-degree variance is {var_in_degree}')
        print(f'The mean out-degree is {mean_out_degree}')
        print(f'The out-degree variance is {var_out_degree}')

    return np.array([mean_in_degree, var_in_degree, mean_out_degree, var_out_degree])


def compute_clustering(adjacency_matrix=None, default_type=0, print_information=False):
    """
    This is a function to compute the clustering mean and variance
    :param adjacency_matrix: the adjacency matrix
    :param default_type: ID of the default digraph
    :param print_information: Boolean determining if the computed values are printed
    :return: a numpy array with the mean and the variance of the clustering
    """

    # print('f = compute_clustering')

    if adjacency_matrix is None:
        adjacency_matrix = default_digraph(default_type=default_type)

    # Get the number of agents
    num_agents = adjacency_matrix.shape[0]
    clustering = []

    topology = digraph2topology(adjacency_matrix=adjacency_matrix)

    for id_agent in range(0, num_agents):
        # Find the set of in-neighbours of agent 'id_agent', excluding itself
        in_neighbour_ast_set = np.concatenate((topology[id_agent][:id_agent], topology[id_agent][id_agent+1:]))
        num_in_neighbours = in_neighbour_ast_set.sum()  # :=
        if num_in_neighbours > 1:

            # If there are more in-neighbours than itself
            in_neighbour_ast = in_neighbour_ast_set.nonzero()[0]
            topology_subset = topology.transpose()[in_neighbour_ast].transpose()[in_neighbour_ast]
            number_internal_edges = topology_subset.sum() - num_in_neighbours
            total_number_edges = num_in_neighbours*(num_in_neighbours-1)
            clustering.append(number_internal_edges/total_number_edges)

        else:

            if in_neighbour_ast_set.sum() == 1:
                # If the agent has a single in-neighbour
                clustering.append(1)
            else:
                # If the only in-neighbour is itself, then append a nan
                clustering.append(np.nan)

    clustering = np.array(clustering)

    mean_clustering = np.nanmean(clustering)
    var_clustering = np.nanvar(clustering)

    if print_information:
        print(f'The mean clustering is {mean_clustering}')
        print(f'The clustering variance is {var_clustering}')

    return np.array([mean_clustering, var_clustering])







