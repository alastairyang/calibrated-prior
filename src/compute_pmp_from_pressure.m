function pmp = compute_pmp_from_pressure(P)
    %compute pressure melting point
    rho_i = 917.0;
    g = 9.81;
    beta=9.8e-8;
    pmp = 273.15 - beta .* P;
end