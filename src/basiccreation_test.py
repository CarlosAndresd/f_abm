from basiccreation import *
import numpy as np


def test_create_random_numbers():
	random_numbers = create_random_numbers()
	assert np.max(abs(random_numbers)) < 1.0