"""

    This is the module that takes care of all the plotting, whether it is for opinion distributions (histograms), or
    digraphs, or whatever it is required.


"""


import numpy as np


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