function chi_2_residual = Na_modeled_by_layer_chi_2_residual(Tm, Na_obs, Na_obs_uncert, beta, Hp, ssCl, NH4p, depth_bound_decim)
    % Tm is depth avearged temperature for the column
    % H+, Cl-, NH4+ concentrations for each layer
    % and depth of the bonnds of each layer
    % MacGregor et al. 2015, equation (8)

    Na_modeled = 0;

    for i = 1:size(Hp, 2)
        molar_Hp = Hp(i);
        molar_ssCl = ssCl(i);
        molar_NH4p = NH4p(i);
    
        sigma = temperature_to_conductivity(Tm, 'W97', beta, molar_Hp, molar_ssCl, molar_NH4p);

        Na_modeled_layer = conductivity_to_atten_rate(sigma);
    
        % weight of each layer is its thickness
        Na_modeled = Na_modeled + Na_modeled_layer * (depth_bound_decim(:, i+1) - depth_bound_decim(:, i));
    end
    % divide by total thickness to get weighted mean,
    % which is our modeled Na
    Na_modeled = Na_modeled/(depth_bound_decim(end) - depth_bound_decim(1));

    chi_2_residual = ((Na_obs - Na_modeled)^2) / (Na_obs_uncert^2);
end