"""

==============================================================
Model Functions, (:mod:`f_abm.src.model_functions`)
==============================================================

Description
-----------

    This module contains all the model related functions, it includes all the functions to execute models, as well as
    functions to execute any model

Functions
_________

    - model_evolution
    - cb_model_step

"""


import numpy as np
from .basic_creation import (create_random_numbers, )
from .digraph_creation import (default_digraph, )


def model_evolution(initial_opinions=None, adjacency_matrix=None, agent_parameters=None, model_parameters=None,
                    model_function=None, num_steps=50, default_type=0):
    """

    This function evolves a given model, with the give initial opinions, adjacency matrix, agent parameters, model
    parameters, and number of steps

    Parameters
    ----------
    initial_opinions: numpy list of initial opinions. By default, it calls the function 'create_opinions()'
    adjacency_matrix: numpy 2d adjacency matrix.
    agent_parameters: agent parameters, what this is depends on the model. By default, it is '[[0.33, 0.33]]*100'
    model_parameters: model parameters, what this is depends on the model. By default, it is '[0.4, 2, 5]'
    model_function: function that evolves the steps of the model. By default, it is 'cb_model_step', i.e. it evolves
                    the Classification-based model
    num_steps: prediction horizon, it is an integer. By default, it is 50
    default_type: ID of the default digraph

    Returns
    -------
    A 2d numpy array with as many rows as agents, and as many columns as num_steps. Each row contains the opinion
    evolution of every agent.

    """

    if initial_opinions is None:
        initial_opinions = create_random_numbers()

    if adjacency_matrix is None:
        adjacency_matrix = default_digraph(default_type=default_type)

    if agent_parameters is None:
        agent_parameters = [[0.33, 0.33]]*100

    if model_parameters is None:
        model_parameters = [0.4, 2, 5]

    if model_function is None:
        model_function = cb_model_step

    # Get the number of agents
    num_agents = initial_opinions.shape[0]

    # Create the 2d array which will store the opinions
    all_opinions = np.zeros((num_agents, num_steps))
    all_opinions[:, 0:1] = initial_opinions
    # start_time = time.time()
    for id_col in range(0, num_steps-1):
        # ini_time = time.time()
        all_opinions[:, (id_col+1):(id_col+2)] = model_function(all_opinions[:, id_col], adjacency_matrix,
                                                                agent_parameters, model_parameters)
        # print(f'{time.time() - ini_time} seconds, iteration {id_col}')
    # print("--- %s seconds ---" % (time.time() - start_time))
    return all_opinions


def cb_model_step(initial_opinions, adjacency_matrix, agent_parameters, model_parameters=(0.4, 2, 5)):
    """

    This function takes a step with the Classification-based model

    Parameters
    ----------
    initial_opinions: a list (or numpy array) of initial conditions
    adjacency_matrix: a list of lists representing the adjacency matrix
    agent_parameters: a list of lists containing the agent parameters, the first parameter is alpha and the second one
                        is beta
    model_parameters: the parameter tuple lambda, xi, and mu

    Returns
    -------

    """

    # Get the number of agents
    num_agents = initial_opinions.shape[0]

    # get the model parameters
    lambda_value = model_parameters[0]  # Opinion change magnitude
    xi_value = model_parameters[1]  # Conformist parameter
    mu_value = model_parameters[2]  # Radical parameter

    # Create the array with new opinions
    new_opinions = np.zeros((num_agents, 1))

    # Auxiliary vectorized function
    # bool2int_vect_func = np.vectorize(boolean2int, o types=[float]) # remove the space between 'o' and 'types'

    # Model Thresholds
    # model_thresholds = [[6/5, 2],  # Thr, # of neighbours that agent $i$ perceives as agreeing much less that itself
    #                     [2/5, 6/5],  # Thr, # of neighbours that agent $i$ perceives as agreeing less that itself
    #                     [-2/5, 2/5],  # Thr, # of neighbours that agent $i$ perceives as agreeing the same that itself
    #                     [-6/5, -2/5],  # Thr, # of neighbours that agent $i$ perceives as agreeing more that itself
    #                     [-2, -6/5]]  # Thr, # of neighbours that agent $i$ perceives as agreeing much more that itself

    # Compute the new opinions for each agent
    for id_agent in range(0, num_agents):

        # Although this implementation is more fancy ....
        # Compute the number of neighbours
        # num_neighbours = bool2int_vect_func(adjacency_matrix[id_agent] != 0.0).sum()

        # Get the neighbour's perceived opinions
        # neighbour_perceived_opinions = [adjacency_matrix[id_agent][i]*initial_opinions[i] for i, element
        #                                 in enumerate(adjacency_matrix[id_agent]) if element != 0.0]

        # shift the opinions to be relative to the current agent
        # opinion_difference = initial_opinions[id_agent] - neighbour_perceived_opinions

        # Compute the number of neighbours in each subset
        # num_elements = np.zeros(5)
        # for id_subset in range(0, 5):
        #     num_elements[id_subset] = (bool2int_vect_func(opinion_difference >= model_thresholds[id_subset][0])
        #                                * bool2int_vect_func(opinion_difference <
        #                                model_thresholds[id_subset][1])).sum()

        # Compute the opinion change
        # opinion_change = (lambda_value / num_neighbours) \
        #                  * ((agent_parameters[id_agent][0] * xi_value * (num_elements[4] - num_elements[0]))
        #                   + (agent_parameters[id_agent][0] * (num_elements[3] - num_elements[1]))
        #                   + (agent_parameters[id_agent][1] * mu_value * num_elements[2] * initial_opinions[id_agent]))

        # ... this one is ~2.7 times faster with 100 agents and 100 time steps. Maybe it is faster in other cases
        num_d_p = 0
        num_d = 0
        num_n = 0
        num_a = 0
        num_a_p = 0
        num_neighbours = 0
        for id_neigh in range(0, num_agents):
            if adjacency_matrix[id_agent][id_neigh] != 0.0:
                num_neighbours += 1
                opinion_difference = initial_opinions[id_agent] \
                    - (adjacency_matrix[id_agent][id_neigh]*initial_opinions[id_neigh])

                if (opinion_difference >= 1.2) and (opinion_difference <= 2.0):
                    num_d_p += 1
                elif (opinion_difference >= 0.4) and (opinion_difference <= 1.2):
                    num_d += 1
                elif (opinion_difference >= -0.4) and (opinion_difference <= 0.4):
                    num_n += 1
                elif (opinion_difference >= -1.2) and (opinion_difference <= -0.4):
                    num_a += 1
                elif (opinion_difference >= -2.0) and (opinion_difference <= -1.2):
                    num_a_p += 1
                else:
                    print('this should not happen')

        # Compute the opinion change
        opinion_change = (lambda_value/num_neighbours) \
            * ((agent_parameters[id_agent][0]*xi_value*(num_a_p-num_d_p))
               + (agent_parameters[id_agent][0]*(num_a-num_d))
               + (agent_parameters[id_agent][1]*mu_value*num_n*initial_opinions[id_agent]))

        new_opinions[id_agent] = np.maximum(np.minimum(initial_opinions[id_agent]+opinion_change, 1.0), -1.0)

    return new_opinions














