from unittest import TestCase
from bxd.core.bound import Bound
from bxd.core.box import Box
import numpy as np

class TestBox(TestCase):
    def test_ensure_norm_consistency(self):
        lower_bound = Bound([1, 0], [1, 0])
        upper_bound = Bound([1, 0], [2, 0])

        box = Box(lower_bound, upper_bound)
        box.lower_bound = lower_bound
        box.upper_bound = upper_bound

        bound_changed = box.ensure_norm_consistency()
        self.assertFalse(bound_changed)
        if box.lower_bound != lower_bound:
            self.fail()
        if box.upper_bound != upper_bound:
            self.fail()


