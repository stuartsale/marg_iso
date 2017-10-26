import numpy


class posterior_cluster:
    """ A class to store clusters in posterior space
    """

    def __init__(self, data, probs):
        """ __init__(data, probs)

            Initialise a cluster in posterior space.

            Parameters
            ----------
            data : ndarray(float)
                The coordinates of the data points associated
                with the cluster
            probs : ndarray(float)
                The probabilities of each of the data points
        """

        self.data = data
        self.probs = probs

        self.set_weight()

    def __len__(self):
        """ __len__()

            Gives the number of points in the cluster

            Returns
            -------
            The number of points in the cluster
        """
        return self.data.shape[0]

    def set_weight(self, weight=None):
        """ set_weight(weight=None)

            Sets the probability weight of the cluster. If no
            weight is provided, the weight is set to the mean
            of the probabilities of each point in the cluster
            multiplied by the standard deviation of the cluster
            member positions (with a floor).

            Parameters
            ----------
            weight : float
                The probaility weight of the cluster
        """

        if weight:
            self.weight = weight
        else:
            self.weight = (np.mean(np.exp(self.probs))
                           * max(np.std(self.data[:, 1]), 0.01)
                           * max(np.std(self.data[:, 2]), 0.01))
