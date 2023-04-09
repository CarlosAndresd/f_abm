import matplotlib.pyplot as plt
import numpy as np
from plot_functions import plot_histogram


def modify_opinions_method_1(opinions, des_mean, des_abs_mean, epsilon=0.05, max_counter=100, show_process=False,
                             limits=(-1, 1)):
    """ This function modifies a given opinion distribution to create an opinion distribution with the desired mean
    and absolute value mean using method 1

    :param opinions: the original opinions
    :param des_mean: the desired opinion mean
    :param des_abs_mean: the desired opinion mean absolute value
    :param epsilon: the tolerance for the infinity norm
    :param max_counter: the maximum number of iterations to find the desired opinions
    :param show_process: boolean determining whether to show the creation process or not
    :param limits: a tuple with the upper and lower limits of the opinions
    :return: the new, modified opinions
    """

    if show_process:
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax2.plot([0, 1, 1, 0], [0, -1, 1, 0])
        ax2.plot(des_abs_mean, des_mean, 's', linewidth=2, color=(0.5, 0.1, 0.2))
        ax2.grid()

    opinion_mean = opinions.mean()
    opinion_abs_mean = np.absolute(opinions).mean()

    mean_difference = np.absolute(opinion_mean - des_mean)
    mean_abs_difference = np.absolute(opinion_abs_mean - des_abs_mean)

    switching_control = 0
    counter = 0
    while ((mean_difference > epsilon) or (mean_abs_difference > epsilon)) and (counter < max_counter):
        counter += 1

        if switching_control == 0:
            switching_control = 1
            # Primary mean modification (vertical)
            xi_value = np.minimum(mean_difference, 0.5)
            if opinion_mean > des_mean:
                # Reduce the opinion mean
                opinions -= xi_value
                for _ in range(5):
                    opinions[np.argmax(opinions)] = -opinions[np.argmax(opinions)]
            else:
                # Increase the opinion mean
                opinions += xi_value
                for _ in range(5):
                    opinions[np.argmin(opinions)] = -opinions[np.argmin(opinions)]
        else:
            switching_control = 0
            # Primary abs mean modification (horizontal)
            mu_value = np.minimum(mean_abs_difference, 0.3)
            if opinion_abs_mean > des_abs_mean:
                # Reduce the opinion abs mean
                opinions -= np.sign(opinions) * mu_value
                opinions *= (1-mu_value)

            else:
                # Increase the opinion abs mean
                opinions += np.sign(opinions) * mu_value
                opinions *= (1+mu_value)

        # Truncate the opinion values
        opinions = np.maximum(np.minimum(opinions, limits[1]), limits[0])

        if show_process:
            ax1.clear()
            plot_histogram(ax1, opinions)
            ax2.plot(opinion_abs_mean, opinion_mean, 'o', linewidth=2, color=(0.2, 0.5, 0.1))
            plt.gcf().canvas.draw()
            plt.pause(0.01)

        opinion_mean = opinions.mean()
        opinion_abs_mean = np.absolute(opinions).mean()

        mean_difference = np.absolute(opinion_mean - des_mean)
        mean_abs_difference = np.absolute(opinion_abs_mean - des_abs_mean)

    return opinions


def modify_opinions_method_2(opinions, des_mean, des_abs_mean, epsilon=0.05, max_counter=100, show_process=False,
                             limits=(-1, 1)):
    """ This function modifies a given opinion distribution to create an opinion distribution with the desired mean
    and absolute value mean using method 2

    :param opinions: the original opinions
    :param des_mean: the desired opinion mean
    :param des_abs_mean: the desired opinion mean absolute value
    :param epsilon: the tolerance for the infinity norm
    :param max_counter: the maximum number of iterations to find the desired opinions
    :param show_process: boolean determining whether to show the creation process or not
    :param limits: a tuple with the upper and lower limits of the opinions
    :return: the new, modified opinions
    """

    if show_process:
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax2.plot([0, 1, 1, 0], [0, -1, 1, 0])
        ax2.plot(des_abs_mean, des_mean, 's', linewidth=2, color=(0.5, 0.1, 0.2))
        ax2.grid()

    opinion_mean = opinions.mean()
    opinion_abs_mean = np.absolute(opinions).mean()

    mean_difference = np.absolute(opinion_mean - des_mean)
    mean_abs_difference = np.absolute(opinion_abs_mean - des_abs_mean)

    switching_control = 0
    counter = 0
    while ((mean_difference > epsilon) or (mean_abs_difference > epsilon)) and (counter < max_counter):
        counter += 1

        if switching_control == 0:
            switching_control = 1
            # Primary mean modification (vertical)

            m1 = (des_mean + 1) / (opinion_mean + 1)
            m2 = (des_mean - 1) / (opinion_mean - 1)
            for count, old_opinion in enumerate(opinions):
                if old_opinion <= opinion_mean:
                    opinions[count] = ((old_opinion + 1)*m1) - 1
                else:
                    opinions[count] = ((old_opinion - 1) * m2) + 1
        else:
            switching_control = 0
            # Primary abs mean modification (horizontal)
            mu_value = np.minimum(mean_abs_difference, 0.3)
            if opinion_abs_mean > des_abs_mean:
                # Reduce the opinion abs mean by contracting the opinions
                # remove the mean
                local_opinion_mean = opinions.mean()
                opinions -= local_opinion_mean
                # contract the opinions
                opinions *= (1-mu_value)
                # restore the mean
                opinions += local_opinion_mean

            else:
                # Increase the opinion abs mean by expanding the opinions
                # remove the mean
                local_opinion_mean = opinions.mean()
                opinions -= local_opinion_mean
                # expand the opinions
                opinions *= (1 + mu_value)
                # restore the mean
                opinions += local_opinion_mean

        # Truncate the opinion values
        opinions = np.maximum(np.minimum(opinions, limits[1]), limits[0])

        if show_process:
            ax1.clear()
            plot_histogram(ax1, opinions)
            ax2.plot(opinion_abs_mean, opinion_mean, 'o', linewidth=2, color=(0.2, 0.5, 0.1))
            plt.gcf().canvas.draw()
            plt.pause(0.01)

        opinion_mean = opinions.mean()
        opinion_abs_mean = np.absolute(opinions).mean()

        mean_difference = np.absolute(opinion_mean - des_mean)
        mean_abs_difference = np.absolute(opinion_abs_mean - des_abs_mean)

    return opinions