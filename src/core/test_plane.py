from unittest import TestCase
import core.bound
import numpy as np
import math


class TestPlane(TestCase):
    def test_hessian_normal_form(self):
        norm = np.array([1, 1])
        point = 2

        plane = core.bound.Plane(norm, point)

        length = math.sqrt(norm[0] + norm[1])
        norm /= length
        point /= length
        if not np.allclose(norm, plane.norm): self.fail("Plane not normalized correctly.")
        self.assertAlmostEquals(point, plane.D)
