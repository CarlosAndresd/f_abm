"""

==============================================================
Command Line Interface, (:mod:`f_abm.src.command_line_interface`)
==============================================================

Description
-----------

    This module contains all the functions to use the program from the command line

Functions
---------

    - read_positive_integer
    - get_parameter_value
    - none_2_default
    - create_new_simulations
    - read_user_input


"""


from ast import literal_eval
from .model_functions import model_evolution, cb_model_step
from .auxiliary_functions import (modify_opinions_method_1, modify_opinions_method_2, create_random_numbers, modify_mean,
                                                    make_row_stochastic, matrix2digraph)
import numpy as np
from .digraph_creation import complete_digraph, ring_digraph, small_world_digraph, random_digraph
from .plot_functions import plot_histogram, plot_digraph, plot_opinions
from datetime import datetime
from os import mkdir as make_new_directory
from os.path import isdir as is_directory


def read_positive_integer(message, default_input):
	"""

	This function is used to read a positive integer, it is used to ask for the number of agents and the number of time
	steps of the simulation

	Parameters
	----------
	message: message to be displayed to the user
	default_input: default input (in text form)

	Returns
	-------
	the positive integer provided by the user

	Reference
	---------

	This function was modified from the answer in
	https://stackoverflow.com/questions/26198131/check-if-input-is-positive-integer

	"""

	while True:
		number = input(message + ' [' + default_input + ']: ')
		while not number:
			number = default_input
		try:
			val = int(number)
			if val < 0:  # if not a positive int print message and ask for input again
				print("Sorry, input must be a positive integer, try again")
				continue
			break
		except ValueError:
			print("This is not an integer number, try again")

	return int(number)


def get_parameter_value(all_parameters, parameter_name):
	"""

	This function receives two strings, the string 'all_parameters' contains parameters and parameter values separated
	by a semicolon, the string 'parameter_name' contains the name of the parameter. The function returns the evaluation
	of the variable related to the first appearance of the parameter name. See the examples.

	Parameters
	----------
	all_parameters: string containing all the parameter names and parameter values, separated by semicolons
	parameter_name: string containing the name of the parameter

	Returns
	-------
	The first value of the parameter if it is contained in the string 'all_parameters', if it is not contained, then it
	returns None

	Examples
	--------

	all_parameters = 'par_rep=(0.2, 0.3, 0.5); par_tol=0.2; print=True'
	parameter_name = 'par_rep'

	returns (0.2, 0.3, 0.5)

	all_parameters = 'par_rep=(0.2, 0.3, 0.5); par_tol=0.2; print=True'
	parameter_name = 'model_name'

	returns None because 'model_name' is not in all_parameters

	all_parameters = 'par_rep=(0.2, 0.3, 0.5); par_tol=0.2; par_tol=0.5; print=True'
	parameter_name = 'par_tol'

	returns 0.5, because in its first appearance the value 0.2 was given instead of the value 0.5


	"""

	first_index = all_parameters.find(parameter_name)
	if first_index == -1:
		return None
	first_part = all_parameters[first_index:]
	last_index = first_part.find(';')
	if last_index == -1:
		parameter_value = first_part[first_part.find('=')+1:]
	else:
		parameter_value = first_part[first_part.find('=') + 1:first_part.find(';')]

	return literal_eval(parameter_value)


def none_2_default(variable, default_value):
	"""

	This is an auxiliary function, it takes a variable and a default value, if the value of the variable is None, then
	it returns the default value

	Parameters
	----------
	variable: the value of the input variable
	default_value: the default value

	Returns
	-------
	The default value if the initial value is None, otherwise, it does not change anything

	"""
	if variable is None:
		return default_value
	else:
		return variable


def create_new_simulations():
	"""

	This is the function in charge of creating new simulations using user inputs from the command line interface

	Returns
	-------
	Nothing

	"""
	simulation_parameters = read_user_input()

	file_name = simulation_parameters['file_name']
	root_directory_name = simulation_parameters['directory_name']

	if not is_directory(root_directory_name):
		make_new_directory('./' + root_directory_name)

	make_new_directory('./' + root_directory_name + '/' + file_name)

	complete_name = './' + root_directory_name + '/' + file_name + '/' + file_name

	# Now that all the information is gathered, we can proceed to execute the simulation
	num_agents = simulation_parameters['num_ag']

	# 1. Create the initial opinions
	print('\n\n\tCreating initial opinions')
	abs_mean_op, mean_op = simulation_parameters['io_loc']
	io_tolerance = simulation_parameters['io_tol']
	io_method = simulation_parameters['io_met']
	io_initial_distribution = simulation_parameters['io_dis']
	io_print = simulation_parameters['io_prt']

	initial_opinions = create_random_numbers(num_agents=num_agents, number_parameters=io_initial_distribution)

	if io_method is None:

		rng = np.random.default_rng()  # this is for the random numbers creation

		if rng.random(1)[0] > 0.5:
			initial_opinions = modify_opinions_method_1(initial_opinions, des_mean=mean_op,
											des_abs_mean=abs_mean_op, epsilon=io_tolerance)

		else:
			initial_opinions = modify_opinions_method_2(initial_opinions, des_mean=mean_op,
											des_abs_mean=abs_mean_op, epsilon=io_tolerance)

	else:

		if io_method == 1:
			initial_opinions = modify_opinions_method_1(initial_opinions, des_mean=mean_op,
														des_abs_mean=abs_mean_op, epsilon=io_tolerance)

		else:
			if not io_method == 2:
				print("The method should be a number between 1 and 2")
			initial_opinions = modify_opinions_method_2(initial_opinions, des_mean=mean_op,
														des_abs_mean=abs_mean_op, epsilon=io_tolerance)

	if io_print:
		plot_histogram(ax=None, opinions=initial_opinions, num_bins=10, histogram_title='Initial Opinions',
					   close_figure=True, file_name=complete_name + "_io_histogram.png")

	print('\tInitial opinions created')
	# 2. Create the adjacency matrix

	print('\n\n\tCreating adjacency matrix')
	dig_lab = simulation_parameters['dig_lab']

	dig_res = simulation_parameters['dig_res']  # row_stochastic
	dig_per = simulation_parameters['dig_per']  # positive_edge_ratio
	dig_tsi = simulation_parameters['dig_tsi']  # topology_signature
	dig_cpr = simulation_parameters['dig_cpr']  # change_probability
	dig_rpr = simulation_parameters['dig_rpr']  # reverse_probability
	dig_bpr = simulation_parameters['dig_bpr']  # bidirectional_probability
	dig_rei = simulation_parameters['dig_rei']  # num_random_edges_it
	dig_epr = simulation_parameters['dig_epr']  # edge_probability

	dig_res = none_2_default(dig_res, False)
	dig_per = none_2_default(dig_per, 1.0)
	dig_cpr = none_2_default(dig_cpr, 0.0)
	dig_rpr = none_2_default(dig_rpr, 0.0)
	dig_bpr = none_2_default(dig_bpr, 0.0)
	dig_rei = none_2_default(dig_rei, 0)
	dig_epr = none_2_default(dig_epr, 0.5)

	dig_prt = simulation_parameters['dig_prt']

	if dig_lab == 'cd':  # Complete digraph
		adjacency_matrix = complete_digraph(num_agents=num_agents, row_stochastic=dig_res, positive_edge_ratio=dig_per,
											print_text=True)

	elif dig_lab == 'gr':  # Generalised ring
		adjacency_matrix = ring_digraph(num_agents=num_agents, topology_signature=dig_tsi, row_stochastic=dig_res,
					 positive_edge_ratio=dig_per, num_random_edges_it=dig_rei, print_text=True)

	elif dig_lab == 'sw':  # Small-world
		adjacency_matrix = small_world_digraph(num_agents=num_agents, topology_signature=dig_tsi, row_stochastic=dig_res,
							positive_edge_ratio=dig_per, change_probability=dig_cpr, reverse_probability=dig_rpr,
							bidirectional_probability=dig_bpr, num_random_edges_it=dig_rei, print_text=True)

	elif dig_lab == 'rd':  # Random digraph
		adjacency_matrix = random_digraph(num_agents=num_agents, row_stochastic=dig_res, positive_edge_ratio=dig_per,
										  edge_probability=dig_epr, print_text=True)

	else:
		print("The selected digraph topology does not exits")
		adjacency_matrix = complete_digraph(num_agents=num_agents, row_stochastic=dig_res, positive_edge_ratio=dig_per,
											print_text=True)

	if dig_prt:
		plot_digraph(digraph=matrix2digraph(adjacency_matrix), file_name=complete_name + "_digraph.png",
					 visual_style=None, close_figure=True)

	print('\tAdjacency matrix created')

	# 3. Create agent parameters

	print('\n\n\tCreating agent parameters')

	w1_des, w2_des, w3_des = simulation_parameters['par_rep']
	par_tol = simulation_parameters['par_tol']
	par_dis = simulation_parameters['par_dis']
	par_prt = simulation_parameters['par_prt']

	par_dis = none_2_default(par_dis, [[0, -1.0, 1.0, 1]])
	par_tol = none_2_default(par_tol, 0.05)

	weights_1 = create_random_numbers(num_agents=num_agents, number_parameters=par_dis, limits=(0, 1))
	weights_1 = modify_mean(weights_1, w1_des, max_counter=15, epsilon=par_tol, limits=(0, 1))

	weights_2 = create_random_numbers(num_agents=num_agents, number_parameters=par_dis, limits=(0, 1))
	weights_2 = modify_mean(weights_2, w2_des, max_counter=15, epsilon=par_tol, limits=(0, 1))

	weights_3 = create_random_numbers(num_agents=num_agents, number_parameters=par_dis, limits=(0, 1))
	weights_3 = modify_mean(weights_3, w3_des, max_counter=15, epsilon=par_tol, limits=(0, 1))

	inner_traits = np.concatenate((weights_1, weights_2, weights_3), axis=1)
	make_row_stochastic(inner_traits)
	inner_traits = np.maximum(np.minimum(inner_traits, 1), 0)
	inner_traits = inner_traits[:, 0:2]

	print('\tAgent parameters created')

	# 4. Run the model

	print('\n\n\tRunning the model')

	mod_lab = simulation_parameters['mod_lab']
	mod_par = simulation_parameters['mod_par']

	if mod_lab == 'CB':
		model_evolution_function = cb_model_step
	else:
		print('The selected mode does not exist')
		model_evolution_function = cb_model_step

	num_ts = simulation_parameters['num_ts']

	opinion_evolution = model_evolution(initial_opinions=initial_opinions, adjacency_matrix=adjacency_matrix,
										agent_parameters=inner_traits, model_parameters=mod_par,
										model_function=model_evolution_function, num_steps=num_ts, default_type=0)

	plot_opinions(opinion_evolution, inner_traits, mod_lab, axes=None, close_figure=True,
				  file_name=complete_name + "_opinion_evolution.png")

	plot_histogram(ax=None, opinions=opinion_evolution[:, -1], num_bins=10, histogram_title='Final Opinions',
				   close_figure=True, file_name=complete_name + "_fo_histogram.png")

	print('\tSimulation complete')
	print('\n' * 5)


def help_input(input_text, question_number):

	help_dictionary = {1: 'Help entry 1',
					   2: 'Help entry 2',
					   3: 'Help entry 3',
					   4: 'Help entry 4',
					   5: 'Help entry 5',
					   6: 'Help entry 6',
					   7: 'Help entry 7',
					   8: 'Help entry 8',}

	if input_text == 'help':
		print(help_dictionary[question_number])
		return True

	elif input_text == '&':
		return True

	else:
		return False


def read_user_input():
	"""

	This function reads all the necessary input from the user to create a new simulation and returns a dictionary with
	all the information for the simulation(s) to happen

	Returns
	-------
	A dictionary with information on how to run the desired simulation. The dictionary contains the following
	information:

	"""

	welcome_text = 'Creation of a new simulation'
	print('\n'*10)
	print('|' + '*' * (2+len(welcome_text)) + '|')
	print('| ' + welcome_text + ' |')
	print('|' + '*' * (2 + len(welcome_text)) + '|')
	print(' ')

	simulation_data = dict()

	# File name
	default_input = datetime.today().strftime('Simulation-%Y%m%d%H%M%S')

	file_name = '&'
	while help_input(file_name, 1):
		file_name = input('1. Enter name of the new simulation [' + default_input + ']: ')
		while not file_name:
			file_name = default_input

	simulation_data['file_name'] = file_name

	# Directory name
	default_input = 'simulation_results'

	directory_name = '&'
	while help_input(directory_name, 2):
		directory_name = input('2. Enter directory where results are saved [' + default_input + ']: ')
		while not directory_name:
			directory_name = default_input

	simulation_data['directory_name'] = directory_name

	# Number of agents
	# Options:
	# - number of agents (num_ag)
	default_input = '100'

	num_ag = '&'
	message = '3. Enter number of agents'
	while help_input(num_ag, 3):
		num_ag = read_positive_integer(message, default_input)

	simulation_data['num_ag'] = num_ag

	# Initial opinion characterisation
	# Options:
	# - location (io_loc)
	# - tolerance (io_tol) [optional]
	# - method (io_met) [optional]
	# - initial_distribution (io_dis) [optional]
	# - print histogram (io_prt) [optional]

	# default_input = 'io_loc=(0.5, 0.1); io_dis=[[0, -1.0, 1.0, 1]]; io_prt=True'
	default_input = 'io_loc=(0.5, 0.1); io_prt=True'

	initial_opinion_char = '&'
	while help_input(initial_opinion_char, 4):
		initial_opinion_char = input('4. Enter initial opinion characterisation [' + default_input + ']: ')

	initial_opinion_char = initial_opinion_char + '; ' + default_input

	simulation_data['io_loc'] = get_parameter_value(initial_opinion_char, 'io_loc')
	simulation_data['io_tol'] = get_parameter_value(initial_opinion_char, 'io_tol')
	simulation_data['io_met'] = get_parameter_value(initial_opinion_char, 'io_met')
	simulation_data['io_dis'] = get_parameter_value(initial_opinion_char, 'io_dis')
	simulation_data['io_prt'] = get_parameter_value(initial_opinion_char, 'io_prt')

	# Model
	# Options:
	# - model label (mod_lab)
	# - model parameters (mod_par) [optional]
	default_input = 'mod_lab="CB"'

	model_id = '&'
	while help_input(model_id, 5):
		model_id = input('5. Enter model [' + default_input + ']: ')

	model_id = model_id + '; ' + default_input

	simulation_data['mod_lab'] = get_parameter_value(model_id, 'mod_lab')
	simulation_data['mod_par'] = get_parameter_value(model_id, 'mod_par')

	# Agent parameter characterisation
	# Options:
	# - parameter representation (par_rep)
	# - tolerance (par_tol) [optional]
	# - initial distribution (par_dis) [optional]
	# - print histogram or alternative representation (par_prt) [optional]
	default_input = 'par_rep=(0.2, 0.3, 0.5); par_prt=True'

	agent_parameter_char = '&'
	while help_input(agent_parameter_char, 6):
		agent_parameter_char = input('6. Enter agent parameter characterisation [' + default_input + ']: ')

	agent_parameter_char = agent_parameter_char + '; ' + default_input

	simulation_data['par_rep'] = get_parameter_value(agent_parameter_char, 'par_rep')
	simulation_data['par_tol'] = get_parameter_value(agent_parameter_char, 'par_tol')
	simulation_data['par_dis'] = get_parameter_value(agent_parameter_char, 'par_dis')
	simulation_data['par_prt'] = get_parameter_value(agent_parameter_char, 'par_prt')

	# Underlying digraph characterisation
	# Options:
	# - digraph label (dig_lab)
	# - digraph parameters (depending on the type) (dig_par) [optional]
	# - print digraph representation (dig_prt) [optional]
	default_input = 'dig_lab="sw"; dig_tsi=[0, 1, 1, 1]; dig_cpr=0.5; dig_prt=True'

	underlying_digraph_char = '&'
	while help_input(underlying_digraph_char, 7):
		underlying_digraph_char = input('7. Enter underlying digraph characterisation [' + default_input + ']: ')

	underlying_digraph_char = underlying_digraph_char + '; ' + default_input

	simulation_data['dig_lab'] = get_parameter_value(underlying_digraph_char, 'dig_lab')
	# simulation_data['dig_par'] = get_parameter_value(underlying_digraph_char, 'dig_par')

	simulation_data['dig_res'] = get_parameter_value(underlying_digraph_char, 'dig_res')  # row_stochastic
	simulation_data['dig_per'] = get_parameter_value(underlying_digraph_char, 'dig_per')  # positive_edge_ratio
	simulation_data['dig_tsi'] = get_parameter_value(underlying_digraph_char, 'dig_tsi')  # topology_signature
	simulation_data['dig_cpr'] = get_parameter_value(underlying_digraph_char, 'dig_cpr')  # change_probability
	simulation_data['dig_rpr'] = get_parameter_value(underlying_digraph_char, 'dig_rpr')  # reverse_probability
	simulation_data['dig_bpr'] = get_parameter_value(underlying_digraph_char, 'dig_bpr')  # bidirectional_probability
	simulation_data['dig_rei'] = get_parameter_value(underlying_digraph_char, 'dig_rei')  # num_random_edges_it
	simulation_data['dig_epr'] = get_parameter_value(underlying_digraph_char, 'dig_epr')  # edge_probability

	simulation_data['dig_prt'] = get_parameter_value(underlying_digraph_char, 'dig_prt')

	# Number of steps
	# Options:
	# - number of time steps (num_ts)
	default_input = '50'

	num_ts = '&'
	message = '8. Enter number of time-step'
	while help_input(num_ts, 1):
		num_ts = read_positive_integer(message, default_input)

	simulation_data['num_ts'] = num_ts

	print('\n'*5)
	# print(simulation_data)

	return simulation_data


if __name__ == '__main__':
	create_new_simulations()
