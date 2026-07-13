import unittest

from bilby.gw.lensing.conversion import (
    bilby_to_gwfast_params,
    get_lensed_parameter_sets,
    gpc_to_mpc,
    mpc_to_gpc,
)


class ConversionTest(unittest.TestCase):
    def test_distance_conversion(self):
        self.assertAlmostEqual(gpc_to_mpc(1.0), 1000.0)
        self.assertAlmostEqual(mpc_to_gpc(1000.0), 1.0)

    def test_bilby_to_gwfast_params(self):
        bilby = dict(
            mass_1=36.0, mass_2=29.0, luminosity_distance=1000.0,
            theta_jn=1.5, a_1=0.3, a_2=0.5, phase=2.0, psi=1.0,
            log10_M_lz=4.0, R_orbit=50.0, src_pos=0.5,
        )
        gwfast = bilby_to_gwfast_params(bilby)
        self.assertAlmostEqual(gwfast['dL'], 1.0)
        self.assertAlmostEqual(gwfast['M_lz'], 1e4)

    def test_get_lensed_parameter_sets_delta_form(self):
        params = dict(
            mass_1=36.0, mass_2=29.0, luminosity_distance=1000.0,
            theta_jn=1.5, phase=2.0, psi=1.0, a_1=0.3, a_2=0.5,
            delta_iota=0.1, delta_phase=0.2, delta_psi=0.0,
            relative_distance=1.5, relative_mass=1.01, delta_time=0.01,
        )
        p1, p2 = get_lensed_parameter_sets(params)
        self.assertAlmostEqual(p2['theta_jn'], p1['theta_jn'] + 0.1)
        self.assertAlmostEqual(p2['luminosity_distance'], p1['luminosity_distance'] * 1.5)


if __name__ == '__main__':
    unittest.main()
