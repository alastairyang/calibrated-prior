
function sigma = temperature_to_conductivity(T, kind, beta, molar_Hp, molar_ssCl, molar_NH4p)
    %{
    Calculate ice conductivity from temperature assuming an Arrhenius relationship.

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.
    kind: string
        W97 or M07 chemistry
    beta: float
        Correction factor for sigma in MacGregor et al. 2015. Only used for W97. Default: 1.
    molar_Hp : float, optional
        Molar concentration of H+ (mol). Default: 2.7e-6.
    molar_ssCl : float, optional
        Molar concentration of ss-Cl (mol). Default: 4.2e-6.
    molar_NH4p : float, optional
        Molar concentration of NH4+ (mol). Default: None.
        Only present in Greenland. If no input, NH4+ will be assumed to be absent.

    Returns
    -------
    sigma : np.ndarray
        Total ice conductivity (S/m).

    References
    ----------
    MacGregor et al. (2015), Table 2.
    """
    %}

    % Physical constants
    k   = 1.380e-23;
    eV  = 1.602176634e-19;
    
    % default concentrations
    % if molar_NH4p is None, assume it is not present
    if nargin < 3
        beta = 1; % no correction
    end
    if nargin < 4
        molar_Hp = 2.7e-6;
    end
    if nargin < 5
        molar_ssCl = 4.2e-6;
    end
    if nargin < 6
        molar_NH4p = 0;
    end
    
    % disp(beta)
    % disp(molar_Hp)
    % disp(molar_ssCl)
    % disp(molar_NH4p)
    
    switch kind
        case "W97"
            Tr  = 258.15; % referene temperature W97
            sigma0 = 9e-6;
            Epure = 0.58 * eV;
            E_Hp = 0.21 * eV;
            E_ssCl = 0.23 * eV;
            E_NH4p = 0.23 * eV;
            mu_Hp = 4;
            mu_ssCl = 0.55;
            mu_NH4p = 1;
        case "M07"
            Tr  = 252.15; % referene temperature M07
            sigma0 = 9.2e-6;
            Epure = 0.51 * eV;
            E_Hp = 0.20 * eV;
            E_ssCl = 0.19 * eV;
            E_NH4p = 0.23 * eV;
            mu_Hp = 3.2;
            mu_ssCl = 0.43;
            mu_NH4p = 0.8;
        otherwise
            error("Invalid chemistry type. Select 'W97' or 'M07'.")
    end
    
    sigma_ice  = sigma0   * exp((Epure  / k) * (1 / Tr - 1.0 ./ T));
    sigma_Hp   = mu_Hp    .* molar_Hp   * exp((E_Hp   / k) * (1 / Tr - 1.0 ./ T));
    sigma_ssCl = mu_ssCl  .* molar_ssCl * exp((E_ssCl / k) * (1 / Tr - 1.0 ./ T));

    sigma_NH4p = mu_NH4p  .* molar_NH4p * exp((E_NH4p / k) * (1 / Tr - 1.0 ./ T));
    
    % disp(sigma_ice)
    % disp(sigma_Hp)
    % disp(sigma_ssCl)
    % disp(sigma_NH4p)
    % disp(sigma_ice + sigma_Hp + sigma_ssCl + sigma_NH4p)
    sigma = (sigma_ice + sigma_Hp + sigma_ssCl + sigma_NH4p) * beta;
