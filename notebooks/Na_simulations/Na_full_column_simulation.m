% configuration

beta = 2.6;
kind = 'W97';

%%
if ismac
    addpath ~/Documents/Glaciology/ISSM-macOS-Silicon-MATLAB/bin/
    addpath ~/Documents/Glaciology/ISSM-macOS-Silicon-MATLAB/lib/
elseif isunix
    addpath ~/Documents/Glaciology/ISSM-Linux-MATLAB/bin/
    addpath ~/Documents/Glaciology/ISSM-Linux-MATLAB/lib/
end

addpath ~/Documents/Glaciology/calibrated-prior/src/

targetFolder = '~/Documents/Glaciology/GrIS-thermal-inference/North_GrIS_results/models/North_GrIS_smb_max_models/ensemble'; 
fileExtension = '*.mat';
searchPattern = fullfile(targetFolder, '**', fileExtension);
fileStruct = dir(searchPattern);
fileStruct = fileStruct(~[fileStruct.isdir]);
md_filelist = fullfile({fileStruct.folder}, {fileStruct.name});

% load Joe's version 2 attenu data
load radar_x_y_attenu_Tm_v2.mat
%%

md_example = loadmodel(md_filelist{1});
md_num_layers = md_example.mesh.numberoflayers;
% NOTE: all models must have the same geometry

depth_md = cell(size(radar_x_y_attenu_Tm_v2, 1), 1);
T_md = cell(size(radar_x_y_attenu_Tm_v2, 1), 1);
Na_full_sim = cell(size(radar_x_y_attenu_Tm_v2, 1), 1);
Na_obs_sim = cell(size(radar_x_y_attenu_Tm_v2, 1), 1);

% used for filtering radar observation points not in domain
table_row_out_of_bound = zeros(size(radar_x_y_attenu_Tm_v2, 1), 1);

%%
% loop through all radar observation points

Na_sim_v2 = radar_x_y_attenu_Tm_v2;
T_md_cell = cell(size(radar_x_y_attenu_Tm_v2, 1), 1);
T_md_cell(:) = {NaN([size(md_filelist, 2), md_num_layers])};
Na_sim_v2.T_md = T_md_cell;

depth_md_cell = cell(size(radar_x_y_attenu_Tm_v2, 1), 1);
depth_md_cell(:) = {NaN([size(md_filelist, 2), md_num_layers])};
Na_sim_v2.depth_md = depth_md_cell;

Na_full_sim_cell = cell(size(radar_x_y_attenu_Tm_v2, 1), 1);
Na_full_sim_cell(:) = {NaN([size(md_filelist, 2), 1])};
Na_sim_v2.Na_full_sim = Na_full_sim_cell;

Na_full_obs_cell = cell(size(radar_x_y_attenu_Tm_v2, 1), 1);
Na_full_obs_cell(:) = {NaN([size(md_filelist, 2), 1])};
Na_sim_v2.Na_obs_sim = Na_full_obs_cell;

for idx_md = 1:size(md_filelist, 2)
    fprintf(['\n loading model ' num2str(idx_md) ' / ' num2str(size(md_filelist, 2))])

    md = loadmodel(md_filelist{idx_md});

    fprintf('\n Progress: 00.00%%');

    for idx_radar = 1:size(radar_x_y_attenu_Tm_v2, 1)
        %disp(['simulating observation ' num2str(idx_radar) ' / ' num2str(size(radar_x_y_attenu_Tm_v2, 1))])
        fprintf('\b\b\b\b\b\b%5.2f%%', (idx_radar/size(radar_x_y_attenu_Tm_v2, 1))*100);
        
        % waitbar(idx_radar/size(radar_x_y_attenu_Tm_v2, 1), f, sprintf('Processing: %d%%', floor((idx_radar/size(radar_x_y_attenu_Tm_v2, 1))*100)));
        % find idx of corresponding on md
        % NOTE: all models must have same geometry
        x_md = md_example.mesh.x;
        y_md = md_example.mesh.y;
        dist = sqrt((x_md - radar_x_y_attenu_Tm_v2.radar_x(idx_radar)).^2 + (y_md - radar_x_y_attenu_Tm_v2.radar_y(idx_radar)).^2);
        
        % find point with closest distance
        % # of points should be same as # of layer
        min_dist = min(dist);
    
        idx_point_on_md = find(dist == min_dist);
    
        if (length(idx_point_on_md) ~= md_num_layers)
            error("check md")
        end
    
        % if closest model point is 100 km away, skip
        if min_dist >= 100 * 1e3
            table_row_out_of_bound(idx_radar) = 1;
            %disp("Radar observation not in model boundary, skipping")
            continue
        end
    
        % for every point in radar data, find its closest point in model

        E_point_on_md = md.results.SteadystateSolution.Enthalpy(idx_point_on_md);
        if any(isnan(E_point_on_md)) == true
            disp(['model ' num2str(idx_radar) ' / ' num2str(size(md_filelist, 1)) ' did not converge, skipping'])
            continue
        end

        pmp_point_on_md = compute_pmp_from_pressure(md.results.SteadystateSolution.Pressure(idx_point_on_md));
        
        T_point_on_md = enthalpy_to_temperature(E_point_on_md, pmp_point_on_md);
        depth_point_on_md = max(md.mesh.z(idx_point_on_md)) - md.mesh.z(idx_point_on_md);
        
        Na_sim_v2.T_md{idx_radar}(idx_md, :) = flipud(T_point_on_md)';
        Na_sim_v2.depth_md{idx_radar}(idx_md, :) = flipud(depth_point_on_md)';
        
        %%

        % interpolate modeled temperature to depths of radar observations

        Hp_sample = radar_x_y_attenu_Tm_v2.Hp(idx_radar);
        Hp_sample = Hp_sample{1};
        ssCl_sample = radar_x_y_attenu_Tm_v2.ssCl(idx_radar);
        ssCl_sample = ssCl_sample{1};
        NH4p_sample = radar_x_y_attenu_Tm_v2.NH4p(idx_radar);
        NH4p_sample = NH4p_sample{1};
        depth_bound_decim_sample = radar_x_y_attenu_Tm_v2.depth_bound_decim(idx_radar);
        depth_bound_decim_sample = depth_bound_decim_sample{1};
        
        T_md_sample = Na_sim_v2.T_md{idx_radar}(idx_md, :);
        %disp(T_md_sample)
        depth_md_sample = Na_sim_v2.depth_md{idx_radar}(idx_md, :);
        
        % spline interpolate to depth bounds of chemistry,
        % append first and last element of modeled temperature
        % (surface and base)
        T_full_column = [T_md_sample(1) spline(depth_md_sample, T_md_sample, depth_bound_decim_sample) T_md_sample(end)];
        
        % get full depth profile, append first and last element of modeled depth
        % (surface and base)
        depth_full_column = [depth_md_sample(1) depth_bound_decim_sample depth_md_sample(end)];
        
        Hp_full_column = [Hp_sample(1) Hp_sample Hp_sample(end)];
        ssCl_full_column = [ssCl_sample(1) ssCl_sample ssCl_sample(end)];
        NH4p_full_column = [NH4p_sample(1) NH4p_sample NH4p_sample(end)];

        %%
        % calculate thickness averaged attenuation rate
        % simulate based on full column depth
        
        Na_full_sim_layer_sum = 0;
        for idx_layer = 1:size(Hp_full_column, 2)
            molar_Hp_layer = Hp_full_column(idx_layer);
            molar_ssCl_layer = ssCl_full_column(idx_layer);
            molar_NH4p_layer = NH4p_full_column(idx_layer);
            T_layer = (T_full_column(idx_layer) + T_full_column(idx_layer+1)) / 2;
            thick_layer = depth_full_column(idx_layer+1) - depth_full_column(idx_layer);
        
            sigma_modeled_layer = temperature_to_conductivity(T_layer, kind, beta, ...
                molar_Hp_layer, molar_ssCl_layer, molar_NH4p_layer);
        
            Na_modeled_layer = conductivity_to_atten_rate(sigma_modeled_layer);
        
            % disp("modeled layer Na")
            % disp(Na_modeled_layer)
            % disp("modeled layer thickness")
            % disp(thick_layer)
        
            Na_full_sim_layer_sum = Na_full_sim_layer_sum + thick_layer * Na_modeled_layer;
        
        end
        % divide by total depth
        Na_full_sim_point_md = Na_full_sim_layer_sum / (depth_full_column(end) - depth_full_column(1));
        Na_sim_v2.Na_full_sim{idx_radar}(idx_md) = Na_full_sim_point_md;
        %%
        % simulate what attenuation rate would be
        % if it is only calculated based on the temperature profile
        % within the observation thickness in MacGregor et al. 2015
        
        Na_obs_sim_layer_sum = 0;
        for idx_layer = 2:size(Hp_full_column, 2)-1
            molar_Hp_layer = Hp_full_column(idx_layer);
            molar_ssCl_layer = ssCl_full_column(idx_layer);
            molar_NH4p_layer = NH4p_full_column(idx_layer);
            T_layer = (T_full_column(idx_layer) + T_full_column(idx_layer+1)) / 2;
            thick_layer = depth_full_column(idx_layer+1) - depth_full_column(idx_layer);
        
            sigma_modeled_layer = temperature_to_conductivity(T_layer, kind, beta, ...
                molar_Hp_layer, molar_ssCl_layer, molar_NH4p_layer);
        
            Na_modeled_layer = conductivity_to_atten_rate(sigma_modeled_layer);
            
            % disp("modeled layer Na")
            % disp(Na_modeled_layer)
            % disp("modeled layer thickness")
            % disp(thick_layer)
        
            Na_obs_sim_layer_sum = Na_obs_sim_layer_sum + thick_layer * Na_modeled_layer;
        
        end
        
        % divide by total depth to get weighted average
        Na_obs_sim_point_md = Na_obs_sim_layer_sum / (depth_full_column(end-1) - depth_full_column(2));
        Na_sim_v2.Na_obs_sim{idx_radar}(idx_md) = Na_obs_sim_point_md;
    end

end

% Na_sim_v2(table_row_out_of_bound == 1, :) = [];

save Na_sim_v2.mat Na_sim_v2

Na_sim_v2_df = py.pandas.DataFrame(Na_sim_v2);
Na_sim_v2_df.to_pickle('Na_sim_v2_df.pkl')

% writetable(Na_sim_v2, 'Na_sim_v2.csv')
%%
% example
%{
T = 255;
kind = 'W97';
molar_Hp = 1e-6;
molar_ssCl = 1e-6;
molar_NH4p = 1e-6;
beta = 2.6;
sigma = temperature_to_conductivity(T, kind, beta, molar_Hp, molar_ssCl, molar_NH4p);
Na = conductivity_to_atten_rate(sigma);
%}

