"""

    This is the module that takes care of all the plotting, whether it is for opinion distributions (histograms), or
    digraphs, or whatever it is required.


"""


import numpy as np
import matplotlib.pyplot as plt
import igraph as ig


def plot_histogram(ax, opinions, num_bins=10, histogram_title='Opinions'):
    """ This function creates and plots the histogram for a set of opinions

    :param ax: the axis where the histogram is plotted
    :param opinions: the set of opinions
    :param num_bins: the number of bins of the histogram, by default it is 10
    :param histogram_title: title of the histogram
    :return:
    """

    ax.grid()
    ax.hist(opinions, bins=np.linspace(-1.0, 1.0, num_bins+1), edgecolor='black')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([0, opinions.shape[0]])
    ax.set_title(histogram_title)
    ax.set_axisbelow(True)


def plot_digraph(digraph, file_name=None, visual_style=None):
    """ Function to plot the digraph

    :param digraph: Digraph to be plotted
    :param file_name: string that is the name of the file to be plotted
    :param visual_style: optional visual style

    :return:
    """

    # print('f = plot_digraph')

    # if digraph is None:
    #     digraph = matrix2digraph()

    if visual_style is None:
        # Get the edge weights
        edge_weights = digraph.es["weight"]
        color_dict = {1.0: "blue", -1.0: "red"}
        digraph.es["color"] = [color_dict[edge_weight] for edge_weight in edge_weights]
        visual_style = {"vertex_size": 0.1}

    if file_name is not None:
        ig.plot(digraph, target=file_name + ".pdf", **visual_style)

    fig, ax = plt.subplots()
    ig.plot(digraph, target=ax, **visual_style, layout="circle")
    plt.show()


