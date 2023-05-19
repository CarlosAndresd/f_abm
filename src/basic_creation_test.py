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
from f_abm.src.auxiliary_functions import create_random_numbers


def test_create_random_numbers():
	"""

	Test function for the 'create_random_numbers' function, it checks that the returned values are within the desired
	interval

	"""
	random_numbers = create_random_numbers()
	assert np.max(abs(random_numbers)) < 1.0
