import numpy as np
def mie_scattering(r, f, epsp, epsb):
    """
    Calculate Mie scattering efficiency factors for spherical inclusions
    embedded in a background medium.

    Parameters
    ----------
    r : float
        Particle radius (m).
    f : float
        Frequency (Hz).
    epsp : complex or float
        Particle relative permittivity.
    epsb : complex or float
        Background relative permittivity.

    Returns
    -------
    Es : float
        Scattering efficiency factor.
    Ea : float
        Absorption efficiency factor.
    Ee : float
        Extinction efficiency.
    Eb : float
        Backscattering efficiency.

    References
    ----------
    Ulaby and Long (2014), equations (8.28b), (8.32a), (8.32b), (8.33), (8.34),
    (8.35), (8.36), (8.37), (8.40).
    Original MATLAB implementation by Natalie Wolfenbarger (nswolfen@gmail.com).
    """
    c         = 3e8
    epsb_real = np.real(epsb)

    np_ = np.sqrt(epsp)          # index of refraction of particle
    nb  = np.sqrt(epsb)          # index of refraction of background
    n   = np_ / nb               # relative index of refraction (8.31a)

    lam = c / f
    chi = (2 * np.pi * r / lam) * np.sqrt(epsb_real)   # normalized circumference (8.31b)

    # ------------------------------------------------------------------
    # Inner loop — iterates Mie series until convergence.
    #
    # mode controls which sigma term is accumulated:
    #   'scattering'  → (2l+1)(|a|² + |b|²)          for Es  (8.32a)
    #   'extinction'  → (2l+1) Re(a + b)              for Ee  (8.32b)
    #   'backscatter' → (-1)^l (2l+1)(a - b)          for Eb  (8.40)
    # ------------------------------------------------------------------
    def _mie_loop(mode):
        # Initial Riccati-Bessel recurrence values (8.35a, 8.35b)
        W_1 = np.sin(chi) + 1j * np.cos(chi)
        W_2 = np.cos(chi) - 1j * np.sin(chi)

        # Initial logarithmic derivative A0 (8.37)
        A = 1.0 / np.tan(n * chi)   # cot(x) = 1/tan(x)

        old_sum = 0.0 + 0.0j
        pdiff   = 1.0
        l       = 1

        while pdiff >= 0.001:
            # Riccati-Bessel recurrence (8.34)
            W = (2 * l - 1) / chi * W_1 - W_2

            # Logarithmic derivative recurrence (8.36)
            A = -l / (n * chi) + 1.0 / (l / (n * chi) - A)

            # Mie coefficients (8.33a, 8.33b)
            num_a = (A / n + l / chi) * np.real(W) - np.real(W_1)
            den_a = (A / n + l / chi) * W          - W_1
            a     = num_a / den_a

            num_b = (n * A + l / chi) * np.real(W) - np.real(W_1)
            den_b = (n * A + l / chi) * W          - W_1
            b     = num_b / den_b

            # Accumulate series term according to mode
            if mode == "scattering":
                sigma = (2 * l + 1) * (abs(a) ** 2 + abs(b) ** 2)   # (8.32a)
            elif mode == "extinction":
                sigma = (2 * l + 1) * np.real(a + b)                 # (8.32b)
            elif mode == "backscatter":
                sigma = (-1) ** l * (2 * l + 1) * (a - b)           # (8.40 inner)

            new_sum = old_sum + sigma

            # Convergence check
            if new_sum != 0:
                pdiff = abs((new_sum - old_sum) / new_sum) * 100
            else:
                pdiff = 0.0

            # Advance recurrence
            l   += 1
            W_2  = W_1
            W_1  = W
            old_sum = new_sum

        return new_sum

    # Run the three series
    sum_s = _mie_loop("scattering")
    sum_e = _mie_loop("extinction")
    sum_b = _mie_loop("backscatter")

    # Final efficiency factors
    Es = (2 / chi ** 2) * np.real(sum_s)       # (8.32a)
    Ee = (2 / chi ** 2) * np.real(sum_e)       # (8.32b)
    Eb = (1 / chi ** 2) * abs(sum_b) ** 2      # (8.40)
    Ea = Ee - Es                                # (8.28b)

    return Es, Ea, Ee, Eb

import numpy as np


def compute_imag_permittivity(frequency, conductivity):
    """
    Compute the imaginary part of permittivity from conductivity.

    Parameters
    ----------
    frequency : float or array-like
        Frequency (Hz).
    conductivity : float or array-like
        Electrical conductivity (S/m).

    Returns
    -------
    eps_imag : complex or np.ndarray
        Imaginary permittivity contribution (complex-valued).
    """
    eps0     = 8.854e-12
    omega    = 2 * np.pi * frequency
    eps_imag = -1j * conductivity / (omega * eps0)
    return eps_imag


def compute_atten_rate(phi, r, f, epsb=1.78**2, epsp=1.0):
    """
    Compute Mie-theory attenuation rates for an ensemble of spherical
    particles across a range of volume fractions.

    Parameters
    ----------
    phi : array-like, shape (M,)
        Volume fractions of inclusions.
    r : array-like, shape (Nr,)
        Particle radii (m).
    f : float
        Frequency (Hz).
    epsb : complex or float, optional
        Background relative permittivity. Default: 1.78² (ice).
    epsp : complex or float, optional
        Particle relative permittivity. Default: 1.0 (air).

    Returns
    -------
    Na_s : np.ndarray, shape (Nr, M)
        Scattering attenuation rate (dB/km).
    Na_a : np.ndarray, shape (Nr, M)
        Absorption attenuation rate (dB/km).
    Na_e : np.ndarray, shape (Nr, M)
        Extinction attenuation rate (dB/km).

    Notes
    -----
    Output array layout: rows → particle radii, columns → volume fractions.
    This matches the MATLAB repmat(Qs(:), 1, length(phi)) convention.
    """
    phi = np.asarray(phi, dtype=float).ravel()   # shape (M,)
    r   = np.asarray(r,   dtype=float).ravel()   # shape (Nr,)

    Nr = len(r)

    Es_vec = np.zeros(Nr)
    Ea_vec = np.zeros(Nr)
    Ee_vec = np.zeros(Nr)

    # Compute Mie efficiency factors for each particle radius
    for i in range(Nr):
        Es, Ea, Ee, _ = mie_scattering(r[i], f, epsp, epsb)
        Es_vec[i] = Es
        Ea_vec[i] = Ea
        Ee_vec[i] = Ee

    # Geometric cross-sections and number densities
    V_particle = (4.0 / 3.0) * np.pi * r ** 3          # shape (Nr,)
    N          = phi / V_particle[:, np.newaxis]         # shape (Nr, M)  — broadcast

    # Physical cross-sections per particle  shape (Nr,)
    Qs = Es_vec * np.pi * r ** 2
    Qa = Ea_vec * np.pi * r ** 2
    Qe = Ee_vec * np.pi * r ** 2

    # Broadcast (Nr,) → (Nr, M) to multiply against N
    Qs_array = Qs[:, np.newaxis]                         # shape (Nr, M) via broadcast
    Qa_array = Qa[:, np.newaxis]
    Qe_array = Qe[:, np.newaxis]

    # Bulk attenuation coefficients (m⁻¹), then convert to dB/km
    alpha_s = N * Qs_array                               # shape (Nr, M)
    alpha_a = N * Qa_array
    alpha_e = N * Qe_array

    dB_per_km = 10 * np.log10(np.exp(1)) * 1e3

    Na_s = dB_per_km * alpha_s
    Na_a = dB_per_km * alpha_a
    Na_e = dB_per_km * alpha_e

    return Na_s, Na_a, Na_e
