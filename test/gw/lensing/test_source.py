import unittest

import numpy as np

from bilby.gw.lensing import (
    agn_lensed_binary_black_hole,
    general_lensed_binary_black_hole,
    reference_bilby_injection,
)


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.frequency_array = np.linspace(10, 512, 500)
        self.injection = reference_bilby_injection()

    def _check_waveform(self, waveform):
        self.assertIn('plus', waveform)
        self.assertIn('cross', waveform)
        self.assertTrue(np.all(np.isfinite(waveform['plus'])))
        self.assertTrue(np.all(np.isfinite(waveform['cross'])))

    def test_agn_lensed_waveform(self):
        wf = agn_lensed_binary_black_hole(self.frequency_array, **self.injection)
        self._check_waveform(wf)

    def test_generic_lensed_waveform(self):
        from bilby.gw.lensing import convert_agn_to_generic_lensed, generic_gwfast_to_bilby_lensed
        generic = convert_agn_to_generic_lensed(self.injection)
        params = generic_gwfast_to_bilby_lensed(generic, self.injection)
        wf = general_lensed_binary_black_hole(self.frequency_array, **params)
        self._check_waveform(wf)


if __name__ == '__main__':
    unittest.main()
