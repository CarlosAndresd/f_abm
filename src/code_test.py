"""

"""

import matplotlib.pyplot as plt
from src.basiccreation import a_random_initial_opinion_distribution, a_random_inner_trait_assignation, a_random_digraph
from src.plot_functions import (plot_histogram, plot_opinions, plot_digraph)
from src.model_functions import (model_evolution, cb_model_step)
from src.data_analysis_functions import gather_data


def example_1(num_agents=10):

	opinions = a_random_initial_opinion_distribution(num_agents=num_agents)
	fig = plt.figure(figsize=(10, 7))
	ax1 = fig.add_subplot(111)
	plot_histogram(ax1, opinions)
	plt.gcf().canvas.draw()
	plt.show()


def example_2(num_agents=10):

	initial_opinions = a_random_initial_opinion_distribution(num_agents=num_agents)
	agent_parameters = a_random_inner_trait_assignation(num_agents=num_agents)
	adjacency_matrix = a_random_digraph(num_agents=num_agents)
	model_parameters = [0.4, 2, 5]
	opinion_evolution = model_evolution(initial_opinions=initial_opinions, adjacency_matrix=adjacency_matrix,
										agent_parameters=adjacency_matrix, model_parameters=model_parameters,
										model_function=cb_model_step, num_steps=50, default_type=0)

	fig = plt.figure(figsize=(10, 7))
	ax1 = fig.add_subplot(111)
	plot_opinions(opinion_evolution, agent_parameters, cb_model_step, axes=ax1)
	plt.gcf().canvas.draw()
	plt.show()


def example_3():
	plot_digraph()


def example_4():
	gather_data(num_agents=100, num_iterations=20, global_name='Gather_data_example')


