import numpy as np
import math


def sign(x):
    return math.copysign(1, x)


class Box:
    """
    Represents a BXD box.
    """

    @property
    def lower_bound(self):
        """
        Gets the lower bound of the box.

        :return: Bound.
        """
        return self.bounds[0]

    @property.setter
    def lower_bound(self, bound):
        """
        Sets the lower bound of the box.
        :type   bound: Bound.
        :param bound: New lower bound.
        """
        self.bounds[0] = bound

    @property
    def upper_bound(self):
        """
        Gets the upper bound of the box.
        :return: Bound.
        """
        return self.bounds[1]

    @property.setter
    def upper_bound(self, bound):
        """
        Sets the upper bound of the box.
        :type   bound: Bound.
        :param bound: New upper bound.
        """
        self.bounds[1] = bound

    def __init__(self, lower_bound, upper_bound, index):
        """

        :param lower_bound: The 'lower' bound of box
        :param upper_bound: The 'upper' bound of box
        :param index: The index of the box.
        :return:
        """

        assert len(lower_bound) == len(
                upper_bound), "Tried to create a box with unmatched dimensionalities on the boundaries!"

        self.bounds = [lower_bound, upper_bound]
        self.index = index
        self.dimensionality = len(lower_bound)

    def is_point_in_box(self, point):

        assert len(point) == self.dimensionality, "Dimensionality of point does not match dimensionality of boundary!"
        # Compute the signed distance from the plane to rho from each bound
        dist_lower = np.dot(point, self.lower_bound.plane)
        dist_upper = np.dot(point, self.upper_bound.plane)

        """
        There is a potential floating point error here when the distance to
        the plane is very near zero.
        Because of the precision of the plane outputted to file,
        it is possible that the sign of distance to the plane could be incorrect.
        In this case, either an inversion will occur, or the next step will place
        us more firmly in the the next box. This function will return None in this
        case, and the calling function will make a decision.

        UPDATE: I've decided to not implement this, but leave here for reference.

        threshold = 1.0e-06
        if abs(dist_lower) < threshold or abs(dist_upper) < threshold:
            return None
        """

        """
        The sign of the distance tells us which direction from the plane the point
        is. The sign is different between the lower and upper bound distances then
        the point is between the boundaries.
        """
        if sign(dist_lower) != sign(dist_upper):
            return True
        else:
            return False
