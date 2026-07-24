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

%v1 Na rate
% Extract attenuation

% % example
% layer_atten_20120330_03 = cell2mat(atten_rate{19}{5}); % one-way attenuation
% x_20120330_03 = cell2mat(x_pk{19}{5});
% y_20120330_03 = cell2mat(y_pk{19}{5});
% ind_decim_mid_20120330_03 = cell2mat(ind_decim_mid{19}{5});
% x_ind = x_20120330_03(ind_decim_mid_20120330_03)*1000; % convert km to m
% y_ind = y_20120330_03(ind_decim_mid_20120330_03)*1000; % convert km to m
% %ind_decim_mid is the mid-point index of the 1-km blocks over which Joe calculates the attenuation rate

load('~/Documents/Glaciology/common_data/Joe-layer-atten-GrIS/atten_all.mat'); % 25 layers
load('~/Documents/Glaciology/common_data/Joe-layer-atten-GrIS/merge_gris.mat'); % 25 layers

addpath ../../src
%%

radar_x_y_attenu_Tm_cell = table('size', [0,13], ...
    'VariableTypes', {'cell', 'cell', 'cell', 'cell', ...
                        'cell', 'cell', 'cell', 'cell', 'cell', ...
                        'cell', ...
                        'cell', 'cell', 'cell'}, ...
    'VariableNames', {'radar_x', 'radar_y', 'Na', 'Na_uncert', ...
                        'Tm', 'Tm_uncert', 'Hp', 'ssCl', 'NH4p', ...
                        'H_count', ...
                        'thick_decim', 'thick_atten', 'depth_bound_decim'});

%%

for i = 1:size(atten_rate, 2)
    if isempty(atten_rate{i}) == 1
        continue
    end
    disp(i)
    for j = 1:size(atten_rate{i}, 2)
        if isempty(atten_rate{i}{j}) == 1
            continue
        end
        attenu_segment = cell2mat(atten_rate{i}{j});
        attenu_uncert_segment = cell2mat(atten_rate_uncert{i}{j});
        Tm_segment = cell2mat(temp_iso{i}{j});
        Tm_uncert_segment = cell2mat(temp_iso_uncert{i}{j});
        H_segment = horzcat(H_trans{i}{j}{:});
        Cl_segment = horzcat(Cl_trans{i}{j}{:});
        NH4_segment = horzcat(NH4_trans{i}{j}{:});
        %num_decim_segment = num_decim{i}{j};

        H_count_segment = cell2mat(cellfun(@(inner_cell) cellfun(@length, inner_cell), H_trans{i}{j}, 'UniformOutput', false));

        thick_decim_segment = cell2mat(thick_decim{i}{j});
        thick_atten_segment = cell2mat(thick_atten{i}{j});
        depth_bound_decim_segment = horzcat(depth_bound_decim{i}{j}{:});

        x_segment = cell2mat(x_pk{i}{j});
        y_segment = cell2mat(y_pk{i}{j});
        ind_decim_mid_segment = cell2mat(ind_decim_mid{i}{j});
        x_ind_segment = x_segment(ind_decim_mid_segment)*1000; % convert km to m
        y_ind_segment = y_segment(ind_decim_mid_segment)*1000; % convert km to m

        %disp(size(num_decim_segment))
        
        for k = 1:size(x_ind_segment, 2)
            if isnan(attenu_segment(k)) == false
            radar_x_y_attenu_Tm_cell{end+1, :} = {x_ind_segment(k)' y_ind_segment(k)' attenu_segment(k)' ...
                attenu_uncert_segment(k)' Tm_segment(k)' Tm_uncert_segment(k)' ...
                H_segment{k}' Cl_segment{k}' NH4_segment{k}' H_count_segment(k)' ...
            thick_decim_segment(k)' thick_atten_segment(k)' depth_bound_decim_segment{k}'};
            end
        end
        % radar_x_y_attenu_Tm_cell = rmmissing(radar_x_y_attenu_Tm_cell);
        % radar_x_y_attenu_Tm = radar_x_y_attenu_Tm(~isnan(radar_x_y_attenu_Tm(:,3)), :);
    end
end

save radar_x_y_attenu_Tm_v1_cell.mat radar_x_y_attenu_Tm_cell
%%

radar_x_y_attenu_Tm_v1 = table('size', [size(radar_x_y_attenu_Tm_cell)], ...
    'VariableTypes', {'double', 'double', 'double', 'double', ...
                        'double', 'double', 'cell', 'cell', 'cell', ...
                        'double', ...
                        'double', 'double', 'cell'}, ...
    'VariableNames', {'radar_x', 'radar_y', 'Na', 'Na_uncert', ...
                        'Tm', 'Tm_uncert', 'Hp', 'ssCl', 'NH4p', ...
                        'H_count', ...
                        'thick_decim', 'thick_atten', 'depth_bound_decim'});

radar_x_y_attenu_Tm_v1.radar_x  = cell2mat(radar_x_y_attenu_Tm_cell.radar_x);
radar_x_y_attenu_Tm_v1.radar_y  = cell2mat(radar_x_y_attenu_Tm_cell.radar_y);
radar_x_y_attenu_Tm_v1.Na  = cell2mat(radar_x_y_attenu_Tm_cell.Na);
radar_x_y_attenu_Tm_v1.Na_uncert  = cell2mat(radar_x_y_attenu_Tm_cell.Na_uncert);
radar_x_y_attenu_Tm_v1.Tm  = cell2mat(radar_x_y_attenu_Tm_cell.Tm);
radar_x_y_attenu_Tm_v1.Tm_uncert  = cell2mat(radar_x_y_attenu_Tm_cell.Tm_uncert);
radar_x_y_attenu_Tm_v1.Hp  = radar_x_y_attenu_Tm_cell.Hp;
radar_x_y_attenu_Tm_v1.ssCl  = radar_x_y_attenu_Tm_cell.ssCl;
radar_x_y_attenu_Tm_v1.NH4p  = radar_x_y_attenu_Tm_cell.NH4p;
radar_x_y_attenu_Tm_v1.H_count  = cell2mat(radar_x_y_attenu_Tm_cell.H_count);
radar_x_y_attenu_Tm_v1.thick_decim  = cell2mat(radar_x_y_attenu_Tm_cell.thick_decim);
radar_x_y_attenu_Tm_v1.thick_atten  = cell2mat(radar_x_y_attenu_Tm_cell.thick_atten);
radar_x_y_attenu_Tm_v1.depth_bound_decim  = radar_x_y_attenu_Tm_cell.depth_bound_decim;


save radar_x_y_attenu_Tm_v1.mat radar_x_y_attenu_Tm_v1
%%
Tm_calculated = NaN(size(radar_x_y_attenu_Tm_v1, 1), 1);

beta = 2.6;

for i = 1:size(radar_x_y_attenu_Tm_v1, 1)
    Na_obs_sample = radar_x_y_attenu_Tm_v1.Na(i);
    Na_obs_uncert_sample = radar_x_y_attenu_Tm_v1.Na_uncert(i);
    Hp_sample = radar_x_y_attenu_Tm_v1.Hp(i);
    Hp_sample = Hp_sample{1};
    ssCl_sample = radar_x_y_attenu_Tm_v1.ssCl(i);
    ssCl_sample = ssCl_sample{1};
    NH4p_sample = radar_x_y_attenu_Tm_v1.NH4p(i);
    NH4p_sample = NH4p_sample{1};

    depth_bound_decim_sample = radar_x_y_attenu_Tm_v1.depth_bound_decim(i);
    depth_bound_decim_sample = depth_bound_decim_sample{1};
    
    objective = @(Tm) Na_modeled_by_layer_chi_2_residual(Tm, Na_obs_sample, Na_obs_uncert_sample, ...
                            beta, Hp_sample, ssCl_sample, NH4p_sample, depth_bound_decim_sample);
    
    Tm_calculated(i) = fminsearch(objective, 250);
end

radar_x_y_attenu_Tm_v1.Tm_calculated = Tm_calculated;
%% 

Tm_array = cell2mat(radar_x_y_attenu_Tm_cell.Tm);
x_array = cell2mat(radar_x_y_attenu_Tm_cell.radar_x);
y_array = cell2mat(radar_x_y_attenu_Tm_cell.radar_y);

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

% % radiostratigraphy_file v1
% radiostratigraphy_file = '~/Documents/Glaciology/North_GrIS_data_processing/data/radiostratigraphy/RRRAG4_Greenland_1993_2013_01_age_grid.nc';
% 
% % get coordinate (in km)
% x = 1e3 * double(ncread(radiostratigraphy_file,'x'));
% y = 1e3 * double(ncread(radiostratigraphy_file, 'y'));
% % get age depth
% depth_iso = double(ncread(radiostratigraphy_file,'depth_iso'));
% age_iso =  double(ncread(radiostratigraphy_file,'age_iso'));
% % total ice thickness interpolated from BedMachine
% H = double(ncread(radiostratigraphy_file,'thick'));
% 
% % get fractional thickness for Holocene Last Glacial Period ice
% % holocene
% holocene_ft = depth_iso(:,:,1)./H;
% % LGP
% lgp_ft = (depth_iso(:,:,4)-depth_iso(:,:,1))./H;
% 
% F_holocene = scatteredInterpolant(reshape(x, [], 1), reshape(y, [], 1), reshape(holocene_ft, [], 1), 'natural');
% holocene_f_interp = F_holocene(radar_x_y_attenu_Tm.radar_x, radar_x_y_attenu_Tm.radar_y);
% holocene_f_interp(isnan(holocene_f_interp)) = 1;
% 
% F_LGP = scatteredInterpolant(reshape(x, [], 1), reshape(y, [], 1), reshape(lgp_ft, [], 1), 'natural');
% LGP_f_interp = F_LGP(radar_x_y_attenu_Tm.radar_x, radar_x_y_attenu_Tm.radar_y);
% LGP_f_interp(isnan(LGP_f_interp)) = 0;
% 
% fraction = [reshape(holocene_f_interp, [], 1) reshape(LGP_f_interp, [], 1)];