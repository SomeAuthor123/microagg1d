import unittest

import numpy as np

from microagg1d.cost_maxdist import AdaptedMaxDistCostCalculator, MaxDistCostCalculator
from microagg1d.cost_round import (
    AdaptedRoundDownCostCalculator,
    AdaptedRoundUpCostCalculator,
    RoundDownCostCalculator,
    RoundUpCostCalculator,
)
from microagg1d.cost_sae import (
    AdaptedSAECostCalculator,
    SAECostCalculator,
    calc_sorted_median,
)
from microagg1d.utils_for_test import remove_numba_from_class, restore_to_class

x = np.arange(10, dtype=np.float64)
F_vals = np.zeros(10)


invalid_val = np.inf


class BasicTests(unittest.TestCase):
    def test_invalid_value(self):
        sae_calculators = [
            AdaptedSAECostCalculator(x, 2, F_vals),
            SAECostCalculator(x),
            RoundDownCostCalculator(x),
            AdaptedRoundUpCostCalculator(x, 2, F_vals),
            RoundUpCostCalculator(x),
            AdaptedRoundDownCostCalculator(x, 2, F_vals),
            MaxDistCostCalculator(x),
            AdaptedMaxDistCostCalculator(x, 2, F_vals),
        ]
        for calculator in sae_calculators:
            self.assertAlmostEqual(calculator.calc(10, 1), invalid_val)
            self.assertAlmostEqual(calculator.calc(1, 0), invalid_val)


class BasicTestsNonCompiled(BasicTests):
    def setUp(self):
        self.cleanup = remove_numba_from_class(
            self.__class__.__bases__[0], allowed_packages=["microagg1d"]
        )

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)


class SAETests(unittest.TestCase):
    def test_sorted_median(self):
        self.assertAlmostEqual(calc_sorted_median(x), 4.5)
        self.assertAlmostEqual(calc_sorted_median(x, lb=1), 5)
        # one element case
        self.assertAlmostEqual(calc_sorted_median(x, ub=1), 0)


class SAETestsNonCompiled(SAETests):
    def setUp(self):
        self.cleanup = remove_numba_from_class(
            self.__class__.__bases__[0], allowed_packages=["microagg1d"]
        )

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)


if __name__ == "__main__":
    unittest.main()
