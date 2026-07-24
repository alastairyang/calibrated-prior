% load("/Users/leo/Documents/Glaciology/North_GrIS_data_processing/data/ensemble_data_pixelated/radar_attenu_grid.mat")
% load("/Users/leo/Documents/Glaciology/North_GrIS_data_processing/data/ensemble_data_pixelated/radar_Tm_grid.mat")
% 
% load("/Users/leo/Documents/Glaciology/North_GrIS_data_processing/data/ensemble_data_pixelated/holocene_depth_grid.mat")
% load("/Users/leo/Documents/Glaciology/North_GrIS_data_processing/data/ensemble_data_pixelated/lgp_depth_grid.mat")
% load("/Users/leo/Documents/Glaciology/North_GrIS_data_processing/data/ensemble_data_pixelated/H_grid.mat")
% 
% load("/Users/leo/Documents/Glaciology/North_GrIS_data_processing/data/ensemble_data_pixelated/radar_Tm_grid.mat")
% radar_Tm_grid = radar_Tm_grid + 273.15;
% 
% addpath /Users/leo/Documents/Glaciology/North_GrIS_data_processing/scripts/
%%

load('~/Documents/Glaciology/North_GrIS_data_processing/data/Joe_attenu_v2/atten_all_grn_for_gt.mat')

radar_x_y_attenu_Tm_v2_cell = table('size', [0,13], ...
    'VariableTypes', {'cell', 'cell', 'cell', 'cell', ...
                        'cell', 'cell', 'cell', 'cell', 'cell', ...
                        'cell', ...
                        'cell', 'cell', 'cell'}, ...
    'VariableNames', {'radar_x', 'radar_y', 'Na', 'Na_uncert', ...
                        'Tm', 'Tm_uncert', 'Hp', 'ssCl', 'NH4p', ...
                        'H_count', ...
                      'thick_decim', 'thick_atten', 'depth_bound_decim'});

for i = 1:size(atten_rate, 2)
    if isempty(atten_rate{i}) == 1
        continue
    end
    disp(i)
    attenu_segment = atten_rate{i};
    attenu_uncert_segment = atten_rate_uncert{i};
    Tm_segment = temp_iso{i};
    Tm_uncert_segment = temp_iso_uncert{i};
    H_segment = horzcat(H_seg{i});
    Cl_segment = horzcat(Cl_seg{i});
    NH4_segment = horzcat(NH4_seg{i});

    H_count_segment = cellfun(@length, H_seg{i});

    thick_decim_segment = thick_decim{i};
    thick_atten_segment = thick_atten{i};
    depth_bound_decim_segment = horzcat(depth_bound_decim{i});

    x_segment = x{i};
    y_segment = y{i};
    ind_decim_mid_segment = ind_decim_mid{i};
    x_ind_segment = x_segment(ind_decim_mid_segment); % in meters
    y_ind_segment = y_segment(ind_decim_mid_segment); % in meters

    for k = 1:size(x_ind_segment, 2)
        if isnan(attenu_segment(k)) == false
        radar_x_y_attenu_Tm_v2_cell{end+1, :} = {x_ind_segment(k)' y_ind_segment(k)' attenu_segment(k)' ...
            attenu_uncert_segment(k)' Tm_segment(k)' Tm_uncert_segment(k)' ...
            H_segment{k}' Cl_segment{k}' NH4_segment{k}' H_count_segment(k)' ...
        thick_decim_segment(k)' thick_atten_segment(k)' depth_bound_decim_segment{k}'};
        end
    end
end
%%

radar_x_y_attenu_Tm_v2 = table('size', [size(radar_x_y_attenu_Tm_v2_cell)], ...
    'VariableTypes', {'double', 'double', 'double', 'double', ...
                        'double', 'double', 'cell', 'cell', 'cell', ...
                        'double', ...
                        'double', 'double', 'cell'}, ...
    'VariableNames', {'radar_x', 'radar_y', 'Na', 'Na_uncert', ...
                        'Tm', 'Tm_uncert', 'Hp', 'ssCl', 'NH4p', ...
                        'H_count', ...
                        'thick_decim', 'thick_atten', 'depth_bound_decim'});

radar_x_y_attenu_Tm_v2.radar_x  = cell2mat(radar_x_y_attenu_Tm_v2_cell.radar_x);
radar_x_y_attenu_Tm_v2.radar_y  = cell2mat(radar_x_y_attenu_Tm_v2_cell.radar_y);
radar_x_y_attenu_Tm_v2.Na  = cell2mat(radar_x_y_attenu_Tm_v2_cell.Na);
radar_x_y_attenu_Tm_v2.Na_uncert  = cell2mat(radar_x_y_attenu_Tm_v2_cell.Na_uncert);
radar_x_y_attenu_Tm_v2.Tm  = cell2mat(radar_x_y_attenu_Tm_v2_cell.Tm);
radar_x_y_attenu_Tm_v2.Tm_uncert  = cell2mat(radar_x_y_attenu_Tm_v2_cell.Tm_uncert);
radar_x_y_attenu_Tm_v2.Hp  = radar_x_y_attenu_Tm_v2_cell.Hp;
radar_x_y_attenu_Tm_v2.ssCl  = radar_x_y_attenu_Tm_v2_cell.ssCl;
radar_x_y_attenu_Tm_v2.NH4p  = radar_x_y_attenu_Tm_v2_cell.NH4p;
radar_x_y_attenu_Tm_v2.H_count  = cell2mat(radar_x_y_attenu_Tm_v2_cell.H_count);
radar_x_y_attenu_Tm_v2.thick_decim  = cell2mat(radar_x_y_attenu_Tm_v2_cell.thick_decim);
radar_x_y_attenu_Tm_v2.thick_atten  = cell2mat(radar_x_y_attenu_Tm_v2_cell.thick_atten);
radar_x_y_attenu_Tm_v2.depth_bound_decim  = radar_x_y_attenu_Tm_v2_cell.depth_bound_decim;

save radar_x_y_attenu_Tm_v2_cell.mat radar_x_y_attenu_Tm_v2_cell
%%
radar_x_y_attenu_Tm = radar_x_y_attenu_Tm_v2;
beta = 2.6;

Tm_calculated = NaN(size(radar_x_y_attenu_Tm, 1), 1);

for i = 1:size(radar_x_y_attenu_Tm, 1)
    Na_obs_sample = radar_x_y_attenu_Tm.Na(i);
    Na_obs_uncert_sample = radar_x_y_attenu_Tm.Na_uncert(i);
    Hp_sample = radar_x_y_attenu_Tm.Hp(i);
    Hp_sample = Hp_sample{1};
    ssCl_sample = radar_x_y_attenu_Tm.ssCl(i);
    ssCl_sample = ssCl_sample{1};
    NH4p_sample = radar_x_y_attenu_Tm.NH4p(i);
    NH4p_sample = NH4p_sample{1};

    depth_bound_decim_sample = radar_x_y_attenu_Tm.depth_bound_decim(i);
    depth_bound_decim_sample = depth_bound_decim_sample{1};
    
    objective = @(Tm) Na_modeled_by_layer_chi_2_residual(Tm, Na_obs_sample, Na_obs_uncert_sample, ...
                            beta, Hp_sample, ssCl_sample, NH4p_sample, depth_bound_decim_sample);
    
    Tm_calculated(i) = fminsearch(objective, 250);
end

radar_x_y_attenu_Tm_v2.Tm_calculated = Tm_calculated;
save radar_x_y_attenu_Tm_v2.mat radar_x_y_attenu_Tm_v2
%% 

Tm_array = radar_x_y_attenu_Tm.Tm;
x_array = radar_x_y_attenu_Tm.radar_x;
y_array = radar_x_y_attenu_Tm.radar_y;

rgb = [ ...
    94    79   162
    50   136   189
   102   194   165
   171   221   164
   230   245   152
   255   255   191
   254   224   139
   253   174    97
   244   109    67
   213    62    79
   158     1    66  ] / 255;

diff = Tm_calculated - (Tm_array + 273.15);

subplot(1,2,1)
scatter(x_array, y_array, 5, diff);
colorbar();
colormap(rgb);
clim([-0.1, 0.1])
title("diff")

subplot(1,2,2)
plot(diff)
colorbar();
clim([0, 1])
title("diff")

%%

% holocene_f = holocene_depth_grid ./ H_grid;
% holocene_f(isnan(holocene_f)) = 0;
% lgp_f = (lgp_depth_grid - holocene_depth_grid) ./ H_grid;
% lgp_f(isnan(lgp_f)) = 0;
% fractions = [reshape(holocene_f, [], 1) reshape(lgp_f, [], 1)];
% 
% Tm = attenuRateToTemperatureGrIS(radar_attenu_grid, 'W97', fractions, 2.2);
% save('/Users/leo/Documents/Glaciology/calibrated-prior/data/NGrIS/Tm_matlab.mat', 'Tm');
% 
% %%
% Tm(Tm == 250) = NaN;
% 
% pcolor(Tm - 273.15);
% colorbar();
% clim([-35 -15])

% %%
% radiostratigraphy_file_v1 = '~/Documents/Glaciology/North_GrIS_data_processing/data/radiostratigraphy/RRRAG4_Greenland_1993_2013_01_age_grid.nc';
% 
% % get coordinate (in km)
% x_v1 = double(ncread(radiostratigraphy_file_v1,'x'));
% y_v1 = double(ncread(radiostratigraphy_file_v1, 'y'));
% % get age depth
% depth_iso_v1 = double(ncread(radiostratigraphy_file_v1,'depth_iso'));
% age_iso_v1 =  double(ncread(radiostratigraphy_file_v1,'age_iso'));
% % total ice thickness interpolated from BedMachine
% H = double(ncread(radiostratigraphy_file_v1,'thick'));
% 
% % get fractional thickness for Holocene Last Glacial Period ice
% % holocene
% holocene_ft_v1 = depth_iso_v1(:,:,1)./H;
% % LGP
% lgp_ft_v1 = (depth_iso_v1(:,:,4)-depth_iso_v1(:,:,1))./H;
% 
% 
% %%
% radiostratigraphy_file = '~/Documents/Glaciology/common_data/Radiostratigraphy_and_Age_Structure_Greenland_v2/RRRAG4_Greenland_1993_2019_02_age_grid.nc';
% 
% % get coordinate (in m)
% x = double(ncread(radiostratigraphy_file,'x'));
% y = double(ncread(radiostratigraphy_file,'y'));
% 
% [X, Y] = meshgrid(x, y);
% % get age depth
% depth_iso = double(ncread(radiostratigraphy_file,'depth_iso'));
% depth_iso = permute(depth_iso, [2 1 3:ndims(depth_iso)]);
% 
% age_iso = double(ncread(radiostratigraphy_file,'age_iso'));
% %%
% % total ice thickness interpolated from BedMachine
% % H = double(ncread(radiostratigraphy_file,'thick'));
% BedMachine_path = '~/Documents/Glaciology/common_data/BedMachine_Greenland_v6/BedMachineGreenland-v6.nc';
% x_BedMachine = double(ncread(BedMachine_path, 'x'));
% y_BedMachine = double(ncread(BedMachine_path, 'y'));
% [X_BedMachine, Y_BedMachine] = meshgrid(x_BedMachine, y_BedMachine);
% 
% thick_BedMachine = transpose(double(ncread(BedMachine_path, 'thickness')));
% 
% thick_interp = interp2(X_BedMachine, Y_BedMachine, thick_BedMachine, ...
%                    X, Y);
% 
% holocene_ft = depth_iso(:,:,4)./thick_interp;
% % LGP
% lgp_ft = (depth_iso(:,:,10)-depth_iso(:,:,4))./thick_interp;
% 
% %%
% % hold on
% % plot(reshape(holocene_ft_v1, [], 1))
% % plot(reshape(holocene_ft, [], 1))
% 
% %%
% F_holocene = scatteredInterpolant(reshape(X, [], 1), reshape(Y, [], 1), reshape(holocene_ft, [], 1));
% holocene_f_interp = F_holocene(radar_x_y_attenu_Tm.radar_x, radar_x_y_attenu_Tm.radar_y);
% holocene_f_interp(isnan(holocene_f_interp)) = 1;
% 
% F_LGP = scatteredInterpolant(reshape(X, [], 1), reshape(Y, [], 1), reshape(lgp_ft, [], 1));
% LGP_f_interp = F_LGP(radar_x_y_attenu_Tm.radar_x, radar_x_y_attenu_Tm.radar_y);
% LGP_f_interp(isnan(LGP_f_interp)) = 0;
% 
% fraction = [reshape(holocene_f_interp, [], 1) reshape(LGP_f_interp, [], 1)];
% %% 
% Tm = attenuRateToTemperatureGrIS(radar_x_y_attenu_Tm.Na, 'W97', fraction, 2.6);
% diff = Tm - (radar_x_y_attenu_Tm.Tm + 273.15);
% 
% subplot(1,3,1)
% scatter(radar_x_y_attenu_Tm.radar_x, radar_x_y_attenu_Tm.radar_y, [], diff);
% colorbar();
% clim([-5, 5])
% 
% subplot(1,3,2)
% scatter(radar_x_y_attenu_Tm.radar_x, radar_x_y_attenu_Tm.radar_y, [], holocene_f_interp);
% colorbar();
% clim([0, 1])
% 
% subplot(1,3,3)
% scatter(radar_x_y_attenu_Tm.radar_x, radar_x_y_attenu_Tm.radar_y, [], LGP_f_interp);
% colorbar();
% clim([0, 1])