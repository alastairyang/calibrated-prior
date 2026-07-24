function Na = conductivity_to_atten_rate(sigma)
    %{
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
    %}

    c    = 3e8;          % speed of light (m/s)
    eps0 = 8.854e-12;    % permittivity of free space (F/m)
    epsr = 3.17;         % real relative permittivity of ice

    Na = 1000 * (10 * log10(exp(1))) .* sigma / (c * eps0 * sqrt(epsr));