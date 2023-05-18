"""

	This module contains all the testing functions for the basiccreation.py module


"""


import numpy as np
from src.auxiliary_functions import create_random_numbers


def test_create_random_numbers():
	random_numbers = create_random_numbers()
	assert np.max(abs(random_numbers)) < 1.0
