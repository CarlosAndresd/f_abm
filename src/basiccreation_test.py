"""

	This module contains all the testing functions for the basiccreation.py module


"""


from auxiliary_functions import create_random_numbers
import numpy as np


def test_create_random_numbers():
	random_numbers = create_random_numbers()
	assert np.max(abs(random_numbers)) < 1.0
