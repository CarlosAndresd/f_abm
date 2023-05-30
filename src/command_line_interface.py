"""

==============================================================
Command Line Interface, (:mod:`f_abm.src.command_line_interface`)
==============================================================

Description
-----------

    This module contains all the functions to use the program from the command line

Functions
---------

    - create_new_simulations


"""


from ast import literal_eval


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

	return number


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


def create_new_simulations():
	"""

	This function reads all the necessary input from the user to create a new simulation and returns a dictionary with
	all the information for the simulation(s) to happen

	Returns
	-------
	A dictionary with information on how to run the desired simulation. The dictionary contains the following
	information:

	"""

	welcome_text = 'Creation of a new simulation'
	print('|' + '*' * (2+len(welcome_text)) + '|')
	print('| ' + welcome_text + ' |')
	print('|' + '*' * (2 + len(welcome_text)) + '|')
	print(' ')

	# File name
	default_input = 'date'
	file_name = input('Enter name of the new simulation [' + default_input + ']: ')
	while not file_name:
		file_name = default_input

	# Number of agents
	# Options:
	# - number of agents (num_ag)
	default_input = '100'
	message = 'Enter number of agents'
	num_agents = read_positive_integer(message, default_input)

	# Initial opinion characterisation
	# Options:
	# - location (io_loc)
	# - tolerance (io_tol) [optional]
	# - method (io_met) [optional]
	# - print histogram (io_prt) [optional]

	default_input = 'loc=(0.5, 0.1), print=True'
	initial_opinion_char = input('Enter initial opinion characterisation [' + default_input + ']: ')
	while not initial_opinion_char:
		initial_opinion_char = default_input

	# Model
	# Options:
	# - model label (mod_lab)
	# - model parameters (mod_par) [optional]
	default_input = 'model_label=CB'
	model_id = input('Enter model [' + default_input + ']: ')
	while not model_id:
		model_id = default_input

	# Agent parameter characterisation
	# Options:
	# - parameter representation (par_rep)
	# - tolerance (par_tol) [optional]
	# - method (par_met) [optional]
	# - print histogram or alternative representation (par_prt) [optional]
	default_input = 'par_rep=(0.2, 0.3, 0.5), print=True'
	agent_parameter_char = input('Enter agent parameter characterisation [' + default_input + ']: ')
	while not agent_parameter_char:
		agent_parameter_char = default_input

	# Underlying digraph characterisation
	# Options:
	# - digraph label (dig_lab)
	# - digraph parameters (depending on the type) (dig_par) [optional]
	# - print digraph representation (dig_prt) [optional]
	default_input = 'digraph_label=sw, sig=(0, 1, 1, 1), alp=0.5, print=True'
	underlying_digraph_char = input('Enter underlying digraph characterisation [' + default_input + '] ')
	while not underlying_digraph_char:
		underlying_digraph_char = default_input

	# Number of steps
	# Options:
	# - number of time steps (num_ts)
	default_input = '50'
	message = 'Enter number of time-step'
	num_time_steps = read_positive_integer(message, default_input)


if __name__ == '__main__':
	create_new_simulations()
