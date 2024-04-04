import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        M = np.where(cmap==observation, 0.9, 0.1)
        alpha = np.zeros_like(belief)
        pos_dist = np.zeros_like(belief)

        if action[0] == -1:
            alpha[:,1:] = 0.1 * belief[:, 1:]
            alpha[:, 0] = belief[:, 0]
            alpha[:, :-1] += 0.9 * belief[:, 1:]
        elif action[0] == 1:
            alpha[:, :-1] = 0.1 * belief[:, :-1]
            alpha[:, -1] = belief[:, -1]
            alpha[:, 1:] += 0.9 * belief[:, :-1]
        elif action[1] == -1:
            alpha[:-1, :] = 0.1 * belief[:-1, :]
            alpha[-1,:] = belief[-1,:]
            alpha[1:, :] += 0.9 * belief[:-1, :]
        else:
            alpha[1:, :] = 0.1 * belief[1:, :]
            alpha[0,:] = belief[0,:]
            alpha[:-1, :] += 0.9 * belief[1:, :]

        pos_dist = alpha * M
        pos_dist = pos_dist / pos_dist.sum()

        return pos_dist
        


