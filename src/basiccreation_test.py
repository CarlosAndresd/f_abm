from CreationModule import *


def test_create_random_numbers():
	random_numbers = create_random_numbers()
	assert abs(temp_c - expected_result) < 1.0e-6

