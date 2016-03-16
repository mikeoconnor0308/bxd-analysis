import numpy as np


class Plane:
    """
    Represents a (hyper)plane.
    """

    def __init__(self, norm, point):
        """
        Initialises an instance of a Plane with norm (a,b,c,..) and point in the form ax + by + cz + ... + point = 0.
        The plane is converted to Hessian normal form.

        :type norm: array_like
        :type point: array_like
        :param norm: Norm of plane.
        :param point: point at which plane lies.
        """
        if type(norm) is not np.array:
            self.norm = np.array(norm)
        else:
            self.norm = norm
        self.point = point
        self.hessian_normal_form()

    def hessian_normal_form(self):
        """
        Converts plane to Hessian normal form, where the norm is a unit vector.
        """
        length = np.linalg.norm(self.norm)
        self.point /= length
        self.norm /= length

    def compute_point_plane_dist(self, v):
        """
        Computes distance between plane and a point.

        :type v: array_like
        :param v: Specified point
        :return: Perpendicular distance.
        :rtype: float
        """

        if type(v) is not np.array:
            v = np.array(v)
        return np.dot(v, self.norm) + self.point

    @property
    def get_dimensionality(self):
        """
        Gets the dimensionality of the plane.
        :return: Dimensionality
        :rtype: integer
        """
        return self.norm.length()


class Bound:
    """
    Represents a BXD boundary.
    """

    def __init__(self, norm, point):
        """
        Constructor for BXD boundary with norm and point.

        :param norm: Norm of plane
        :param point: Point on plane where trajectory is expected to cross boundary (on average)
        """
        self.plane = Plane(norm, point)
        self.point = point



