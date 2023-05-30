"""

==============================================================
Basic Creation Test, (:mod:`f_abm.src.basic_creation_test`)
==============================================================

Description
-----------

    This module contains all the test functions for the 'basic_creation.py' module

Functions
---------

    - test_create_random_numbers

"""


import numpy as np
from .auxiliary_functions import create_random_numbers
from .command_line_interface import get_parameter_value


def test_create_random_numbers():
	"""

	Test function for the 'create_random_numbers' function, it checks that the returned values are within the desired
	interval

	"""
	random_numbers = create_random_numbers()
	assert np.max(abs(random_numbers)) < 1.0


def test_get_parameter_value_1():
	"""

	Test function for the 'get_parameter_value' function

	Returns
	-------
	Nothing

	"""

	all_parameters = 'par_rep=(0.2, 0.3, 0.5); par_tol=0.2; print=True'
	parameter_name = 'par_rep'
	assert get_parameter_value(all_parameters, parameter_name) == (0.2, 0.3, 0.5)


def test_get_parameter_value_2():
	"""

	Test function for the 'get_parameter_value' function

	Returns
	-------
	Nothing

	"""

	all_parameters = 'par_rep=(0.2, 0.3, 0.5); par_tol=0.2; print=True'
	parameter_name = 'model_name'
	assert get_parameter_value(all_parameters, parameter_name) is None


def test_get_parameter_value_3():
	"""

	Test function for the 'get_parameter_value' function

	Returns
	-------
	Nothing

	"""

	all_parameters = 'par_rep=(0.2, 0.3, 0.5); par_tol=0.2; par_tol=0.5; print=True'
	parameter_name = 'par_tol'
	assert get_parameter_value(all_parameters, parameter_name) == 0.2




