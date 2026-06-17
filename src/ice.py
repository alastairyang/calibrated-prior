import numpy as np
import torch
def compute_pmp(H):
    """  
    compute pressure melting point
    """
    rho_i = 917.0
    g = 9.81
    beta=9.8e-8
    return 273.15 - rho_i * g * H * beta 

def enthalpy_to_temperature(Eb, Tpmp, Cp=2093.0, T0=223.15, istorch=True):
    """  
    Compute temperature from enthalpy

    Tb = Tpmp if Eb > Cp*(Tpmp-T0)
    Tb = (Eb + Cp*T0)/Cp if Eb <= Cp*(Tpmp-T0)

    """

    if istorch:
        Tb = torch.where(Eb > Cp*(Tpmp-T0), Tpmp, (Eb + Cp*T0)/Cp)
    else: # numpy
        Tb = np.where(Eb > Cp*(Tpmp-T0), Tpmp, (Eb + Cp*T0)/Cp)
    return Tb

def enthalpy_to_water_fraction(Eb, Tpmp, Cp=2093.0, T0=223.15, istorch=True):
    """  
    Compute water fraction from enthalpy

    w = 0 if Eb <= Cp*(Tpmp-T0)
    w = (Eb - Cp*(Tpmp-T0))/(L + Cp*(Tpmp-T0)) if Eb > Cp*(Tpmp-T0)

    """
    L = 334000.0  # J/kg
    if istorch:
        w = torch.where(Eb <= Cp*(Tpmp-T0), 0.0, (Eb - Cp*(Tpmp-T0))/L)
    else: # numpy
        w = np.where(Eb <= Cp*(Tpmp-T0), 0.0, (Eb - Cp*(Tpmp-T0))/L)
    return w


def defineActivationEnergies(T):
    """
    Compute activation energies for creep, grain growth, and grain boundary mobility.

    Parameters:
        T : array-like or float — Temperature in Kelvin

    Returns:
        Qg : ndarray — Activation energy for grain growth (J/mol)
        Qc : ndarray — Activation energy for creep (J/mol)
        Qm : ndarray — Activation energy for grain boundary mobility (J/mol)
    """
    T    = np.atleast_1d(np.array(T, dtype=float))
    Temp = T - 273.0  # Convert to Celsius

    # --- Activation Energy for Creep ---
    tp, tm, tc = 0, -20, -10
    Qcp, Qcm   = 100.0, 60.0  # kJ/mol
    c1 = (Qcp - Qcm) / (np.arctan(tp - tc) - np.arctan(tm - tc))
    c2 = Qcp - c1 * np.tanh(tp - tc)

    # --- Activation Energy for Grain Growth ---
    tp, tm, tc = 0, -20, -10
    Qgp, Qgm   = 100.0, 40.0  # kJ/mol
    g1 = (Qgp - Qgm) / (np.arctan(tp - tc) - np.arctan(tm - tc))
    g2 = Qgp - g1 * np.tanh(tp - tc)

    # --- Activation Energy for Grain Boundary Mobility ---
    tp, tm, tc = 0, -20, -10
    Qmp, Qmm   = 40.0, 100.0  # kJ/mol
    m1 = (Qmp - Qmm) / (np.arctan(tp - tc) - np.arctan(tm - tc))
    m2 = Qmp - m1 * np.arctan(tp - tc)

    # --- Compute outputs ---
    Qg = (g1 * np.arctan(Temp - tc) + g2) * 1e3  # J/mol
    Qc = (c1 * np.arctan(Temp - tc) + c2) * 1e3  # J/mol
    Qm = (m1 * np.arctan(Temp - tc) + m2) * 1e3  # J/mol

    return Qg, Qc, Qm


def computeGlenFlowRateParameter(T):
    """
    Compute the Glen flow rate parameter A (Arrhenius-type).

    Parameters:
        T : array-like or float — Temperature in Kelvin

    Returns:
        Aglen : ndarray — Glen flow rate parameter (Pa^-3 s^-1)
    """
    T = np.atleast_1d(np.array(T, dtype=float))

    R = 8.314  # J/(mol·K)

    Qg, Qc, Qm = defineActivationEnergies(T)

    # Reference pre-exponential factor (normalised to 263 K)
    A0 = 2.4e-24 / np.exp(-(115000.0 / R) * ((1.0 / 273.0) - (1.0 / 263.0)))

    Aglen = A0 * np.exp(-(Qc / R) * ((1.0 / T) - (1.0 / 263.0)))

    return Aglen

def driving_stress(alpha, H):
    rho_i = 917  # kg/m³
    g = 9.81     # m/s²
    return rho_i * g * H * alpha

def delta_driving_stress(delta_alpha, H):
    rho_i = 917  # kg/m³
    g = 9.81     # m/s²
    return rho_i * g * H * delta_alpha

def strain_heating(alpha, H, z, n=3, T=250):
    """
    Ice strain heating due to vertical shear deformation.
    
    Parameters:
    alpha : array-like or float — Surface slope (dimensionless)
    H     : array-like or float — Ice thickness (m)
    z     : array-like or float — [0, H] Vertical coordinate (m)
    n     : int — Glen's flow law exponent (default: 3)
    T     : array-like or float — Temperature in Kelvin (default: 250 K)
    """
    rho_i = 917  # kg/m³
    g = 9.81     # m/s²
    A = computeGlenFlowRateParameter(T)
    return 2 *A * (rho_i * g * (H-z) * alpha)**(n+1)

def temperature_to_attenu_rate_ice(T):
    """
    compute attenuation rate from depth-avg temperature (kelvin)
    assuming pure ice
    """
    Tr = 251 # kelvin
    sigma0 = 9.2e-6 # S/m
    E0 = 0.51*1.602e-19 # J (0.51 is in eV)
    k = 1.380e-23 # J/K, Boltzmann's constant
    c = 3e8 # m/s, speed of light
    eps0 = 8.854e-12 # permittivity in free space
    epsr = 3.17 # real relative permittivity

    sigma = sigma0 * np.exp(-(E0/k)*(1/T - 1/Tr))
    attenu_rate = sigma / (c * eps0 * np.sqrt(epsr) / (1000*(10*np.log(np.exp(1)))))

    return attenu_rate

def attenu_rate_to_temperature_ice(Na):
    """
    compute temperature from attenuation rate (Np/m)
    assuming pure ice
    """
    Tr = 251 # kelvin
    sigma0 = 9.2e-6 # S/m
    E0 = 0.51*1.602e-19 # J (0.51 is in eV)
    k = 1.380e-23 # J/K, Boltzmann's constant
    c = 3e8 # m/s, speed of light
    eps0 = 8.854e-12 # permittivity in free space
    epsr = 3.17 # real relative permittivity

    sigma = Na * (c * eps0 * np.sqrt(epsr) / (1000*(10*np.log(np.exp(1)))))
    T = 1 / ((-k/E0)*np.log(sigma/sigma0) + 1/Tr)

    return T

def temperature_to_conductivity(
    T,
    sigma0=6.6e-6,
    Epure=None,
    E_Hp=None,
    E_ssCl=None,
    mu_Hp=3.2,
    mu_ssCl=0.43,
    molar_ssCl=4.2e-6,
    molar_Hp=2.7e-6,
):
    """
    Calculate ice conductivity from temperature assuming an Arrhenius relationship.

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.
    sigma0 : float, optional
        Pure-ice conductivity pre-factor (S/m). Default: 6.6e-6.
    Epure : float, optional
        Activation energy for pure ice (J). Default: 0.55 eV.
    E_Hp : float, optional
        Activation energy for H+ ions (J). Default: 0.20 eV.
    E_ssCl : float, optional
        Activation energy for ss-Cl ions (J). Default: 0.19 eV.
    mu_Hp : float, optional
        Molar conductivity for H+ (S/m per mol/L). Default: 3.2.
    mu_ssCl : float, optional
        Molar conductivity for ss-Cl (S/m per mol/L). Default: 0.43.
    molar_ssCl : float, optional
        Molar concentration of ss-Cl (mol). Default: 4.2e-6.
    molar_Hp : float, optional
        Molar concentration of H+ (mol). Default: 2.7e-6.

    Returns
    -------
    sigma : np.ndarray
        Total ice conductivity (S/m).

    References
    ----------
    MacGregor et al. (2007), Table 1 & 2.
    """
    T = np.asarray(T, dtype=float)

    # Physical constants
    Tr  = 251.0
    k   = 1.380e-23
    eV  = 1.602176634e-19

    if Epure is None:
        Epure = 0.55 * eV
    if E_Hp is None:
        E_Hp = 0.20 * eV
    if E_ssCl is None:
        E_ssCl = 0.19 * eV

    sigma_ice  = sigma0   * np.exp((Epure  / k) * (1 / Tr - 1.0 / T))
    sigma_Hp   = mu_Hp    * molar_Hp   * np.exp((E_Hp   / k) * (1 / Tr - 1.0 / T))
    sigma_ssCl = mu_ssCl  * molar_ssCl * np.exp((E_ssCl / k) * (1 / Tr - 1.0 / T))

    return sigma_ice + sigma_Hp + sigma_ssCl


# ──────────────────────────────────────────────────────────────────────────────
# Helper 2: Conductivity → Attenuation Rate
# ──────────────────────────────────────────────────────────────────────────────

def conductivity_to_atten_rate(sigma):
    """
    Calculate one-way attenuation rate from conductivity.

    Parameters
    ----------
    sigma : float or array-like
        Ice conductivity (S/m).

    Returns
    -------
    N : np.ndarray
        One-way attenuation rate (dB/km).

    References
    ----------
    MacGregor et al. (2007).
    """
    sigma = np.asarray(sigma, dtype=float)

    c    = 3e8          # speed of light (m/s)
    eps0 = 8.854e-12    # permittivity of free space (F/m)
    epsr = 3.17         # real relative permittivity of ice

    N = 1000 * (10 * np.log10(np.exp(1))) * sigma / (c * eps0 * np.sqrt(epsr))

    return N


# ──────────────────────────────────────────────────────────────────────────────
# Combined: Temperature → Attenuation Rate (with chemistry mix)
# ──────────────────────────────────────────────────────────────────────────────

def temperature_to_atten_rate_mix(
    T,
    sigma0=6.6e-6,
    Epure=None,
    E_Hp=None,
    E_ssCl=None,
    mu_Hp=3.2,
    mu_ssCl=0.43,
    molar_ssCl=4.2e-6,
    molar_Hp=2.7e-6,
):
    """
    Calculate one-way radar attenuation rate from a temperature profile,
    accounting for pure-ice conductivity and ionic (H+, ss-Cl) contributions
    via an Arrhenius mixing model.

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.
    sigma0 : float, optional
        Pure-ice conductivity pre-factor (S/m). Default: 6.6e-6.
    Epure : float, optional
        Activation energy for pure ice (J). Default: 0.55 eV.
    E_Hp : float, optional
        Activation energy for H+ ions (J). Default: 0.20 eV.
    E_ssCl : float, optional
        Activation energy for ss-Cl ions (J). Default: 0.19 eV.
    mu_Hp : float, optional
        Molar conductivity for H+ (S/m per mol/L). Default: 3.2.
    mu_ssCl : float, optional
        Molar conductivity for ss-Cl (S/m per mol/L). Default: 0.43.
    molar_ssCl : float, optional
        Molar concentration of ss-Cl (mol). Default: 4.2e-6.
    molar_Hp : float, optional
        Molar concentration of H+ (mol). Default: 2.7e-6.

    Returns
    -------
    N : np.ndarray
        One-way attenuation rate (dB/km).

    References
    ----------
    MacGregor et al. (2007), Table 1 & 2.
    """
    sigma = temperature_to_conductivity(
        T,
        sigma0=sigma0,
        Epure=Epure,
        E_Hp=E_Hp,
        E_ssCl=E_ssCl,
        mu_Hp=mu_Hp,
        mu_ssCl=mu_ssCl,
        molar_ssCl=molar_ssCl,
        molar_Hp=molar_Hp,
    )

    return conductivity_to_atten_rate(sigma)


def temperature_to_rigidity(temperature):
    """
   Rigidity (in s^(1/3) Pa) is the flow law parameter in the flow law
    sigma = B * e^(1/3)  (Cuffey and Paterson, p75).

    Parameters
    ----------
    temperature : float or array-like
        Temperature in Kelvin (must be positive).

    Returns
    -------
    rigidity : np.ndarray
        Rigidity values in s^(1/3) Pa.
    """
    temperature = np.atleast_1d(np.asarray(temperature, dtype=float))

    if np.any(temperature < 0):
        raise ValueError("Input temperature should be in Kelvin (positive).")

    T = temperature - 273.15
    rigidity = np.zeros(T.shape)

    # Piecewise cubic spline segments
    mask = T < -45
    rigidity[mask] = 1e8 * (
        -0.000396645116301 * (T[mask] + 50) ** 3
        + 0.013345579471334 * (T[mask] + 50) ** 2
        - 0.356868703259105 * (T[mask] + 50)
        + 7.272363035371383
    )

    mask = (-45 <= T) & (T < -40)
    rigidity[mask] = 1e8 * (
        -0.000396645116301 * (T[mask] + 45) ** 3
        + 0.007395902726819 * (T[mask] + 45) ** 2
        - 0.253161292268336 * (T[mask] + 45)
        + 5.772078366321591
    )

    mask = (-40 <= T) & (T < -35)
    rigidity[mask] = 1e8 * (
        +0.000408322072669 * (T[mask] + 40) ** 3
        + 0.001446225982305 * (T[mask] + 40) ** 2
        - 0.208950648722716 * (T[mask] + 40)
        + 4.641588833612773
    )

    mask = (-35 <= T) & (T < -30)
    rigidity[mask] = 1e8 * (
        -0.000423888728124 * (T[mask] + 35) ** 3
        + 0.007571057072334 * (T[mask] + 35) ** 2
        - 0.163864233449525 * (T[mask] + 35)
        + 3.684031498640382
    )

    mask = (-30 <= T) & (T < -25)
    rigidity[mask] = 1e8 * (
        +0.000147154327025 * (T[mask] + 30) ** 3
        + 0.001212726150476 * (T[mask] + 30) ** 2
        - 0.119945317335478 * (T[mask] + 30)
        + 3.001000667185614
    )

    mask = (-25 <= T) & (T < -20)
    rigidity[mask] = 1e8 * (
        -0.000193435838672 * (T[mask] + 25) ** 3
        + 0.003420041055847 * (T[mask] + 25) ** 2
        - 0.096781481303861 * (T[mask] + 25)
        + 2.449986525148220
    )

    mask = (-20 <= T) & (T < -15)
    rigidity[mask] = 1e8 * (
        +0.000219771255067 * (T[mask] + 20) ** 3
        + 0.000518503475772 * (T[mask] + 20) ** 2
        - 0.077088758645767 * (T[mask] + 20)
        + 2.027400665191131
    )

    mask = (-15 <= T) & (T < -10)
    rigidity[mask] = 1e8 * (
        -0.000653438900191 * (T[mask] + 15) ** 3
        + 0.003815072301777 * (T[mask] + 15) ** 2
        - 0.055420879758021 * (T[mask] + 15)
        + 1.682390865739973
    )

    mask = (-10 <= T) & (T < -5)
    rigidity[mask] = 1e8 * (
        +0.000692439419762 * (T[mask] + 10) ** 3
        - 0.005986511201093 * (T[mask] + 10) ** 2
        - 0.066278074254598 * (T[mask] + 10)
        + 1.418983411970382
    )

    mask = (-5 <= T) & (T < -2)
    rigidity[mask] = 1e8 * (
        -0.000132282004110 * (T[mask] + 5) ** 3
        + 0.004400080095332 * (T[mask] + 5) ** 2
        - 0.074210229783403 * (T[mask] + 5)
        + 1.024485188140279
    )

    mask = T >= -2
    rigidity[mask] = 1e8 * (
        -0.000132282004110 * (T[mask] + 2) ** 3
        + 0.003209542058346 * (T[mask] + 2) ** 2
        - 0.051381363322371 * (T[mask] + 2)
        + 0.837883605537096
    )

    # Clamp any non-physical negative values
    rigidity[rigidity < 0] = 1e6

    return rigidity


