"""

==============================================================
Functions for Plotting, (:mod:`f_abm.src.plot_functions`)
==============================================================

Description
-----------

    This is the module that takes care of all the plotting, whether it is for opinion distributions (histograms), or
    digraphs, or whatever it is required.

Functions
---------

    - plot_digraph
    - plot_opinions
    - plot_histogram
    - plot_inner_traits
    - plot_all_opinions

"""


import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from .auxiliary_functions import (matrix2digraph, opinion2color, histogram_classification)
from .digraph_creation import default_digraph


def plot_digraph(digraph=None, file_name=None, visual_style=None, close_figure=False, figure_size=(10, 7)):
    """

    Function to plot the digraph

    Parameters
    ----------
    digraph: Digraph to be plotted, by default it is a simple ring digraph
    file_name: string that is the name of the file to be plotted
    visual_style: optional visual style
    close_figure: boolean determining if the figure must be closed
    figure_size: size of the figure to be produced

    Returns
    -------

    """

    if digraph is None:
        digraph = matrix2digraph(default_digraph(default_type=0))

    if visual_style is None:
        # Get the edge weights
        edge_weights = digraph.es["weight"]
        color_dict = {1.0: "blue", -1.0: "red"}
        digraph.es["color"] = [color_dict[edge_weight] for edge_weight in edge_weights]
        visual_style = {"vertex_size": 0.1}

    if file_name is not None:
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)
        ig.plot(digraph, target=ax, **visual_style, layout="circle")
        plt.savefig(fname=file_name, format='png')
        # ig.plot(digraph, target=file_name + ".png", **visual_style, layout="circle")
        # digraph_plot = ig.plot(digraph, **visual_style, layout="circle")
        # digraph_plot.save(file_name + ".png")

    # fig, ax = plt.subplots()
    # ig.plot(digraph, target=ax, **visual_style, layout="circle")
    # plt.show()

    # if close_figure:
    #     print('Close digraph')
    #     plt.close(fig)


def plot_opinions(opinion_evolution, agent_parameters, opinion_model, axes=None, file_name=None, close_figure=False,
                  figure_size=(10, 7)):
    """

    Function to plot the opinion evolution

    Parameters
    ----------
    opinion_evolution: matrix with the opinion evolution data
    agent_parameters: parameters for each agent
    opinion_model: the label of the opinion model
    axes: the axes for the plot
    file_name: string that is the name of the file to be plotted
    close_figure: boolean determining if the figure must be closed
    figure_size: size of the figure to be produced

    Returns
    -------
    Nothing

    """

    # Get the number of agents
    num_agents = opinion_evolution.shape[0]
    num_steps = opinion_evolution.shape[1]
    if axes is None:
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)
    else:
        ax = axes
    for id_agent in range(num_agents):
        ax.plot(opinion_evolution[id_agent], color=opinion2color(opinion_model, agent_parameters[id_agent]))
    ax.set_xlim([0, num_steps])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('Opinion evolution')

    if axes is None:
        plt.grid()
        # display the plot
        # plt.show()
    else:
        ax.grid()

    if file_name is not None:
        plt.savefig(fname=file_name, format='png')

    if close_figure:
        plt.close(fig)


def plot_histogram(ax, opinions, num_bins=10, histogram_title='Opinions', file_name=None, close_figure=False,
                   figure_size=(10, 7)):
    """

    This function creates and plots the histogram for a set of opinions

    Parameters
    ----------
    ax: the axis where the histogram is plotted
    opinions: the set of opinions
    num_bins: the number of bins of the histogram, by default it is 10
    histogram_title: title of the histogram
    file_name: string that is the name of the file to be plotted
    close_figure: boolean determining if the figure must be closed
    figure_size: size of the figure to be produced

    Returns
    -------
    Nothing

    """

    if ax is None:
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)

    ax.grid()
    ax.hist(opinions, bins=np.linspace(-1.0, 1.0, num_bins+1), edgecolor='black')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([0, opinions.shape[0]])
    ax.set_title(histogram_title)
    ax.set_axisbelow(True)

    if file_name is not None:
        plt.savefig(fname=file_name, format='png')

    if close_figure:
        plt.close(fig)


def plot_inner_traits(file_name='standard_inner_traits.npy', figure_size=(10, 7)):
    """

    Function to plot the inner traits for the Classification-based model

    Parameters
    ----------
    file_name: name of the file that contains the inner traits
    figure_size: size of the figure to be produced

    Returns
    -------
    Nothing

    """

    all_inner_traits = np.load(file_name)  # loads your saved array into variable all_opinions
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)

    for inner_traits in all_inner_traits:
        av_con, av_rad = np.maximum(np.minimum(inner_traits.mean(axis=0), 1), 0)
        # Truncation is necessary to avoid problems with negative averages that produce non-existent colours
        # These negative averages may be produced by small numerical errors
        av_stb = 1 - (av_con + av_rad)
        ax.plot(av_con, av_stb, 'o', color=opinion2color(opinion_model='CB', agent_parameter=[av_con, av_rad]))

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('All Inner Traits')
    plt.grid()
    # display the plot
    plt.show()


def plot_all_opinions(file_name='standard_initial_opinions.npy', color_by_type=False, figure_size=(10, 7)):
    """

    Function to plot a set of opinion distributions in the Agreement Plot

    Parameters
    ----------
    file_name: name of the file that contains all the initial opinion distributions
    color_by_type: boolean specifying how to color the plot
    figure_size: size of the figure to be produced

    Returns
    -------
    Nothing

    """
    all_opinions = np.load(file_name)  # loads your saved array into variable all_opinions
    if color_by_type:
        point_colors = [(0.16862745, 0.34901961, 0.76470588),
                        (0.32941176, 0.54901961, 0.18431373),
                        (0.82745098, 0.39607843, 0.50980392),
                        (0.94509804, 0.56078431, 0.00392157),
                        (0.39607843, 0.05098039, 0.10588235)]

    else:
        point_colors = [(0.16862745, 0.54901961, 0.10588235),
                        (0.16862745, 0.54901961, 0.10588235),
                        (0.16862745, 0.54901961, 0.10588235),
                        (0.16862745, 0.54901961, 0.10588235),
                        (0.16862745, 0.54901961, 0.10588235)]

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    ax.plot([0, 1, 1, 0], [0, -1, 1, 0], linewidth=2, color=(0.2, 0.5, 0.8))
    counters = np.zeros((5, 1))
    for opinion_distribution in all_opinions:
        classification = histogram_classification(opinion_distribution)
        counters[classification] += 1
        ax.plot(np.absolute(opinion_distribution).mean(), opinion_distribution.mean(), 'o', linewidth=1.5,
                markersize=3, color=point_colors[classification])
    ax.grid()
    plt.show()

    print(f'number PC = {counters[0]}')
    print(f'number Co = {counters[1]}')
    print(f'number Po = {counters[2]}')
    print(f'number Cl = {counters[3]}')
    print(f'number Di = {counters[4]}')
