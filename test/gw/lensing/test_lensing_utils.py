import unittest

import numpy as np

from bilby.gw.lensing.lensing_utils import (
    convert_y_from_Einstein_to_Rorbit,
    get_agn_lensed_parameters,
)


class LensingUtilsTest(unittest.TestCase):
    def test_convert_y_from_Einstein_to_Rorbit(self):
        y = convert_y_from_Einstein_to_Rorbit(0.99, 50.0)
        self.assertTrue(np.isfinite(y))
        self.assertGreater(abs(y), 0)

    def test_get_agn_lensed_parameters_finite(self):
        params = {
            'Mc': 30.0, 'eta': 0.24, 'iota': 0.99 * np.pi / 2,
            'phase': 2.0, 'psi': 1.0, 'chi1z': 0.3, 'chi2z': 0.5,
            'tcoal': 0.0, 'R_orbit': 50.0, 'M_lz': 1e4, 'src_pos': 0.5,
            'dL': 1.0,
        }
        plus, minus = get_agn_lensed_parameters(params)
        for key in ['Mc', 'dL', 'iota', 'phase']:
            self.assertTrue(np.isfinite(plus[key]))
            self.assertTrue(np.isfinite(minus[key]))
        self.assertNotEqual(plus['phase'], minus['phase'])


if __name__ == '__main__':
    unittest.main()
