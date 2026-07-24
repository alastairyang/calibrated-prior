function pmp = compute_pmp(H)
    %compute pressure melting point
    rho_i = 917.0;
    g = 9.81;
    beta=9.8e-8;
    pmp = 273.15 - rho_i * g .* H * beta;
end