import numpy as np
import numbers


class Plane:
    """
    Represents a (hyper)plane.
    """

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                np.allclose(self.norm, other.norm) and
                self.D == other.D)

    def __init__(self, norm, point):
        """
        Initialises an instance of a Plane with norm (a,b,c,..) and point in the form ax + by + cz + ... + d = 0.
        The plane is converted to Hessian normal form.

        :type norm: array_like
        :type point: float
        :param norm: Norm of plane.
        :param point: point at which plane lies.
        """
        if type(norm) is not np.array:
            self.norm = np.array(norm)
        else:
            self.norm = norm

        # if point passed is a single numeric value, then set D directly
        if isinstance(point, numbers.Number):
            self.D = point
        # otherwise, D from point and norm.
        else:
            point = np.array(point)
            assert len(point) == len(norm), "Plane construction requires either decimal value for distance" \
                                            "or a point on plane of same dimensionality as norm"
            self.D = -np.dot(norm, point)
        self.hessian_normal_form()

    def hessian_normal_form(self):
        """
        Converts plane to Hessian normal form, where the norm is a unit vector.
        """
        length = np.linalg.norm(self.norm)
        self.D /= length
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
        return np.dot(v, self.norm) + self.D

    @property
    def dimensionality(self):
        """
        Gets the dimensionality of the plane.
        :return: Dimensionality
        :rtype: integer
        """
        return len(self.norm)


class Bound:
    """
    Represents a BXD boundary.
    """

    @property
    def dimensionality(self):
        """
        Gets the dimensionality of the boundary
        :return: Dimensionality
        :rtype: integer
        """
        return self.plane.dimensionality

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.plane == other.plane and
                self.point == other.point)

    def __init__(self, norm, point):
        """
        Constructor for BXD boundary with norm and point.

        :param norm: Norm of plane
        :param point: Point on plane where trajectory is expected to cross boundary (on average)
        """
        self.plane = Plane(norm, point)
        self.point = np.array(point)

    def distance(self, point):
        """
        Computes the distance between the Bound and the point.
        :param point: A point in CV space of the Bound
        :return: float.
        """
        assert len(point) == self.dimensionality, "Dimensionality of point does not match that of boundary."
        return self.plane.compute_point_plane_dist(point)
