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

    @lower_bound.setter
    def lower_bound(self, bound):
        """
        Sets the lower bound of the box.
        :type   bound: Bound.
        :param bound: New lower bound.
        """
        self.bounds[0] = bound
        self.ensure_norm_consistency()

    @property
    def upper_bound(self):
        """
        Gets the upper bound of the box.
        :return: Bound.
        """
        return self.bounds[1]

    @upper_bound.setter
    def upper_bound(self, bound):
        """
        Sets the upper bound of the box.
        :type   bound: Bound.
        :param bound: New upper bound.
        """
        self.bounds[1] = bound
        self.ensure_norm_consistency()

    def __init__(self, lower_bound, upper_bound, index=0):
        """

        :type index: int
        :param lower_bound: The 'lower' bound of box
        :param upper_bound: The 'upper' bound of box
        :param index: The index of the box.
        :return:
        """

        assert lower_bound.dimensionality == upper_bound.dimensionality, \
            "Tried to create a box with unmatched dimensionalities on the boundaries!"

        self.bounds = [lower_bound, upper_bound]
        self.index = index
        self.dimensionality = lower_bound.dimensionality

        self.ensure_norm_consistency()

    def ensure_norm_consistency(self):
        """
        Ensures that the norms of the lower and upper bounds are oriented in a direction consistent with the
        functional framework of BXD.

        This is required for determining whether a point is between two bounds (see is_point_in_box).
        """

        bound_changed = False

        # compute signed distance from lower bound to upper bound
        lower_to_upper_dist = self.upper_bound.distance(self.lower_bound.point)
        # this distance should be greater than zero, implying there's a progression from lower to upper bound
        # in the geometry.
        if lower_to_upper_dist < 0:
            self.lower_bound.plane.norm *= -1
            self.lower_bound.plane.D *= -1
            bound_changed = True

        # compute signed distance from upper bound to lower bound
        upper_to_lower_dist = self.lower_bound.distance(self.upper_bound.point)
        if upper_to_lower_dist > 0:
            self.lower_bound.plane.norm *= -1
            self.lower_bound.plane.D *= -1
            bound_changed = True

        return bound_changed

    def is_point_in_box(self, point):

        assert len(point) == self.dimensionality, "Dimensionality of point does not match dimensionality of boundary!"
        # Compute the signed distance from the plane to point from each bound
        dist_lower = self.lower_bound.distance(point)
        dist_upper = self.upper_bound.distance(point)

        """
        The sign of the distance tells us which direction from the plane the point
        is. If the sign is different between the lower and upper bound distances then
        the point is between the boundaries.
        """
        if sign(dist_lower) != sign(dist_upper):
            return True
        else:
            return False
