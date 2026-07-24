function T = enthalpy_to_temperature(E, Tpmp, Cp, T0)

    % Compute temperature from enthalpy
    if nargin <= 2
        Cp = 2093.0;
        T0 = 223.15;
    end

    % T = Tpmp if E > Cp*(Tpmp-T0)
    % T = (E + Cp*T0)/Cp if E <= Cp*(Tpmp-T0)

    T = (E + Cp*T0)/Cp;
    T(E > Cp*(Tpmp-T0)) = Tpmp(E > Cp*(Tpmp-T0));
end