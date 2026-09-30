"""
AGN/CBC lensing utilities ported from gwfast (pure numpy).

Distances inside this module use gwfast conventions: luminosity distance in Gpc.
"""
import numpy as np
from astropy.cosmology import Planck18 as cosmo

from ...core.utils.constants import gravitational_constant, solar_mass, speed_of_light

# gwfast-compatible constants
MRSUN_SI = gravitational_constant * solar_mass / speed_of_light ** 2  # metres
MTSUN_SI = gravitational_constant * solar_mass / speed_of_light ** 3  # seconds
uGpc = 3.085677581491367278913937957796471611e25  # metres
DAY_TO_SEC = 3600.0 * 24.0

# Numerical guards
MIN_WAVEFORM_R_ORBIT = 2.0 # Minimum R_orbit allowed in the prior, prevents velocity blow-up
_EDGE_ON_COS_EPS = 1e-8 # cos(iota) smaller than this sets the waveform to inf
_ARG_CLIP = 1.0 # Trig functions return nan for |x| > 1

zGridGlob = np.logspace(start=-6, stop=5, base=10, num=7000)
dLGridGlob = cosmo.luminosity_distance(zGridGlob).value / 1000.0  # Gpc


def _get_alpha_hat(r_orbit, approx=1):
    """Deflection angle from orbital radius (R_Sch units)."""
    approx_simp = np.sqrt(2 / r_orbit)
    if approx == 1:
        return approx_simp
    if approx == 2:
        idx = -0.5415779752686682
        y0 = -0.6327303836364937
        return (1 + 10 ** y0 * r_orbit ** idx) * approx_simp
    raise ValueError(f"Unknown approx={approx}")


def einstein_angle(lens_mass_source, angular_D_L, D_LS):
    """Einstein radius in radians."""
    rsch = 2 * lens_mass_source * MRSUN_SI
    rsch_2_gpc = rsch / uGpc
    rsch_dl = rsch_2_gpc / angular_D_L
    d_ls = D_LS * rsch_dl
    return np.sqrt(2 * rsch_dl * d_ls / (1 + d_ls))


def get_phi_L(iota, y_src_pos, phi_N):
    sin_iota = np.sin(iota)
    if not np.isfinite(sin_iota) or np.abs(sin_iota) < 1e-15:
        return np.nan
    inside = 1.0 - y_src_pos ** 2
    if inside < 0:
        return np.nan
    arg = np.sqrt(inside) / sin_iota
    if np.abs(arg) > _ARG_CLIP:
        return np.nan
    return phi_N - np.sign(y_src_pos) * np.arccos(arg)


def Keplerian_speed(r_orbit):
    return (2 * r_orbit - 1) ** (-0.5)


def gravitational_redshift(r_orbit):
    return (1 - 1 / r_orbit) ** 0.5 - 1


def Lorentz_factor(beta):
    return (1 - beta ** 2) ** (-0.5)


def PML_image_position(beta_src, theta_E=1):
    _beta = beta_src / theta_E
    sqrt_term = np.sqrt(4 + _beta ** 2)
    img_p = (_beta + sqrt_term) / 2 * theta_E
    img_m = (_beta - sqrt_term) / 2 * theta_E
    return img_p, img_m


def PML_time_delay_magnification(beta_src, theta_E=1):
    img_p, img_m = PML_image_position(beta_src, theta_E)

    _beta = beta_src / theta_E
    sqrt_term = np.sqrt(4 + _beta ** 2)
    delta_t_geom = -_beta * sqrt_term
    delta_t_shap = 2 * np.log(np.abs(img_m / img_p))
    delta_t = (delta_t_geom + delta_t_shap) * 2

    common_term = _beta / sqrt_term + sqrt_term / _beta
    mag_p = 0.25 * (common_term + 2)
    mag_m = 0.25 * (common_term - 2)

    return delta_t, mag_p, mag_m


def line_of_sight_unit_vec(iota, phase):
    phi = np.pi / 2 - phase
    return np.array([
        np.sin(iota) * np.cos(phi),
        np.sin(iota) * np.sin(phi),
        np.cos(iota),
    ])


def convert_y_from_Einstein_to_Rorbit(y_Eins, r_orbit):
    kappa = y_Eins * y_Eins / r_orbit
    return np.sign(y_Eins) * np.sqrt(2 * kappa * (np.sqrt(1 + kappa * kappa) - kappa))


def convert_y_from_Einstein_to_Rorbit_first_order(y_Eins, r_orbit, delta):
    kappa = y_Eins * y_Eins / r_orbit
    delta_r = r_orbit * delta
    surd = np.sqrt(1 + (kappa - delta_r) ** 2)
    t2 = kappa + delta_r
    return np.sign(y_Eins) * np.sqrt(2 * kappa / (surd + t2))


def get_agn_lens_angles(
        redshifted_lens_mass, r_orbit, source_position,
        luminosity_distance, angular_distances=False):
    """Return theta_E (rad), beta (rad), lens_mass_source (M_sun)."""
    d_LS = r_orbit * np.sqrt(1 - source_position ** 2)
    _source_position = source_position * r_orbit

    z = np.interp(luminosity_distance, dLGridGlob, zGridGlob)
    lens_mass_source = redshifted_lens_mass / (1 + z)
    r_sch = 2 * lens_mass_source * MRSUN_SI
    delta = r_sch / uGpc

    if angular_distances:
        ang_lum_dist = luminosity_distance / (1 + z) ** 2
        beta = np.arcsin(_source_position / ang_lum_dist * delta)
        ang_D_S = ang_lum_dist * np.cos(beta)
    else:
        beta = _source_position / luminosity_distance * delta
        ang_D_S = luminosity_distance

    ang_D_L = ang_D_S / (1 + d_LS * delta)
    theta_E = einstein_angle(lens_mass_source, ang_D_L, d_LS)

    return theta_E, beta, lens_mass_source


def _psi_perturbation_term(iota, delta_phi):
    """Equivalent to sin(ι) sqrt(tan²ι + 1/sin²δφ) without tan(ι)."""
    sin_iota = np.sin(iota)
    cos_iota = np.cos(iota)
    sin_dphi = np.sin(delta_phi)
    if np.abs(sin_dphi) < 1e-15:
        return np.nan
    if np.abs(cos_iota) < _EDGE_ON_COS_EPS:
        return np.inf
    return np.sqrt(sin_iota ** 4 / cos_iota ** 2 + sin_iota ** 2 / sin_dphi ** 2)


def compute_lensed_angles_approx(agn_bbh_system_params, angular_distances=False):
    parameters = agn_bbh_system_params.copy()
    iota = parameters["iota"]
    phase = parameters["phase"]
    psi = parameters["psi"]
    r_orbit = parameters["R_orbit"]
    y_src = parameters["src_pos"]

    if not np.isfinite(r_orbit) or r_orbit < MIN_WAVEFORM_R_ORBIT:
        return None
    if np.abs(y_src) > 1.0:
        return None

    v_orb = Keplerian_speed(r_orbit)
    if not np.isfinite(v_orb) or v_orb >= 1.0:
        return None

    phi_N = np.pi / 2 - phase

    theta_E, beta, lens_mass_src = get_agn_lens_angles(
        parameters["M_lz"], r_orbit, y_src, parameters["dL"],
        angular_distances=angular_distances)

    img_pos_1, img_pos_2 = PML_image_position(beta, theta_E)

    alpha_hat = _get_alpha_hat(r_orbit)
    theta_bar_p = alpha_hat - (img_pos_1 - beta)
    theta_bar_m = alpha_hat - (img_pos_2 + beta)

    delta_phi = -get_phi_L(iota, y_src, 0)
    if not np.isfinite(delta_phi):
        return None

    sin_iota = np.sin(iota)
    if np.abs(sin_iota) < 1e-15:
        return None

    inv_delta = (np.cos(iota) ** 2 + sin_iota ** 2 * np.sin(delta_phi) ** 2) ** -0.5
    iota_term = np.cos(iota) * np.cos(delta_phi) * inv_delta
    phi_term = np.sin(delta_phi) / sin_iota * inv_delta
    psi_term = _psi_perturbation_term(iota, delta_phi)
    speed_term = sin_iota * np.cos(delta_phi) * inv_delta

    iota_p = iota - theta_bar_p * iota_term
    iota_m = iota + theta_bar_m * iota_term
    phi_p = phi_N + theta_bar_p * phi_term
    phi_m = phi_N - theta_bar_m * phi_term
    if not np.isfinite(psi_term):
        return None
    if np.isinf(psi_term):
        psi_p = psi
        psi_m = psi
    else:
        psi_p = psi + theta_bar_p / psi_term
        psi_m = psi + theta_bar_m / psi_term

    gamma = Lorentz_factor(v_orb)
    v_proj = -v_orb * sin_iota * np.sin(delta_phi)
    v_orb_p = v_proj * (1 + theta_bar_p * speed_term)
    v_orb_m = v_proj * (1 - theta_bar_m * speed_term)

    z_rel_p = gamma * (1 + v_orb_p) - 1
    z_rel_m = gamma * (1 + v_orb_m) - 1
    z_grav = gravitational_redshift(r_orbit)

    delta_time, mu_p, mu_m = PML_time_delay_magnification(beta, theta_E)
    delta_time *= lens_mass_src * MTSUN_SI
    sqrt_mu_p = np.sqrt(np.abs(mu_p))
    sqrt_mu_m = np.sqrt(np.abs(mu_m))

    out = {
        'iota_p': iota_p,
        'iota_m': iota_m,
        'phase_p': np.pi / 2 - phi_p,
        'phase_m': np.pi / 2 - phi_m,
        'psi_p': psi_p,
        'psi_m': psi_m,
        'v_proj_p': v_orb_p,
        'v_proj_m': v_orb_m,
        'z_rel_p': z_rel_p,
        'z_rel_m': z_rel_m,
        'z_grav': z_grav,
        'delta_time': delta_time,
        'sqrt_mu_p': sqrt_mu_p,
        'sqrt_mu_m': sqrt_mu_m,
    }
    if not all(np.isfinite(v) for v in out.values()):
        return None
    return out


def get_agn_lensed_parameters(unlensed_parameters):
    plus_image_params = unlensed_parameters.copy()
    minus_image_params = unlensed_parameters.copy()

    lensed_params = compute_lensed_angles_approx(unlensed_parameters)
    if lensed_params is None:
        return None, None

    plus_redshift_factor = (1 + lensed_params['z_rel_p']) * (1 + lensed_params['z_grav'])
    minus_redshift_factor = (1 + lensed_params['z_rel_m']) * (1 + lensed_params['z_grav'])

    plus_image_params['iota'] = lensed_params['iota_p']
    plus_image_params['phase'] = lensed_params['phase_p']
    plus_image_params['psi'] = lensed_params['psi_p']
    plus_image_params['Mc'] *= plus_redshift_factor
    plus_image_params['dL'] /= lensed_params['sqrt_mu_p']
    plus_image_params['dL'] *= (1 + lensed_params['z_rel_p']) * plus_redshift_factor
    minus_image_params['iota'] = lensed_params['iota_m']
    minus_image_params['phase'] = lensed_params['phase_m']
    minus_image_params['psi'] = lensed_params['psi_m']
    minus_image_params['Mc'] *= minus_redshift_factor
    minus_image_params['dL'] /= lensed_params['sqrt_mu_m']
    minus_image_params['dL'] *= (1 + lensed_params['z_rel_m']) * minus_redshift_factor
    minus_image_params['tcoal'] += lensed_params['delta_time'] / DAY_TO_SEC

    return plus_image_params, minus_image_params


def convert_simple_PML_to_general_lensed_parameters(parameters):
    output_params = parameters.copy()
    luminosity_distance = output_params.pop("dL")

    theta_E, beta, lens_mass_src = get_agn_lens_angles(
        output_params['M_lz'], output_params['R_orbit'],
        output_params['src_pos'], luminosity_distance,
    )
    time_delay, mag_1, mag_2 = PML_time_delay_magnification(beta_src=beta, theta_E=theta_E)

    output_params['delta_time'] = time_delay * lens_mass_src * MTSUN_SI
    output_params['dL_1'] = luminosity_distance / np.sqrt(np.abs(mag_1))
    output_params['dL_2'] = luminosity_distance / np.sqrt(np.abs(mag_2))
    output_params['delta_iota'] = np.zeros_like(mag_1)
    output_params['delta_phi_12'] = np.zeros_like(mag_1)
    output_params['delta_psi'] = np.zeros_like(mag_1)
    output_params['relative_mass'] = np.ones_like(mag_1)
    return output_params
