function starn_inverse_singlefreq(pred_idx, data_type, input_path)
% starn_inverse_singlefreq solves an inverse problem based on single 
% frequency data. The data is either generated randomly in this script 
% or read from pred.mat
%
% Argument: 
% 
% pred_idx: an integer explicitly determining the case to be solved from 
% 'nn_stored' data or implicitly determining the case sampled from
% sample_fc.m
%
% data_type and input_path:
% There are three options for data_type and input_path should be
% corresponding path to read the needed file.
% 1. data_type = 'nn_stored', input_path should be data path storinig the
%    predicted coefficients on the valid data, like 
%    './data/star10_kh10_n48_2000/valid_predby_pretrained.mat'. The code
%    will use DL pred coefficients to warm-start the Gauss-Newton solver
% 2. data_type = 'random', input_path should be the config path defining
%    the problem, like './configs/nc3.json'. The code will use zeros to
%    initialize the coefficients in the Gauss-Newton solver
% 3. data_type = 'nn', input_path should be path to a saved mode, like 
%    './data/star10_kh10_n48_pretrained/pretrained'. The code will use DL
%    pred coefficients to warm-start the Gauss-Newton solver


close all
clearvars -except pred_idx data_type input_path


test_origin_alg = true;
if strcmp(data_type, 'nn_stored')
    pred_path = input_path;
    nn_pred = load(pred_path);
    cfg_str = nn_pred.cfg_str;
elseif strcmp(data_type, 'random')
    cfg_path = input_path;
    cfg_str = fileread(cfg_path);
    model_name = 'random';
elseif strcmp(data_type, 'nn')
    % nc_test and noise_level are set manually in the script
    model_path = input_path;
    nc_test = 0; % use nc in cfg_path if nc_test=0
    noise_level = 0;
    cfg_path = fullfile(model_path, 'data_config.json');
    cfg_str = fileread(cfg_path);
    tmp_str = split(model_path, '/');
    if model_path(end) == '/'
        model_name = tmp_str{end-1};
        dirname = strjoin(tmp_str(1:end-2), '/');
    else
        model_name = tmp_str{end};
        dirname = strjoin(tmp_str(1:end-1), '/');
    end
    env_path = readlines('env_path.txt');
    env_path = env_path(1); % only read the first line
end
cfg_str = erase(cfg_str, '\n'); % jsondecode cannot read '\n' (in big data)
cfg = jsondecode(cfg_str);
ndata = cfg.ndata;
nc = cfg.nc; % max number of wiggles
n  = max(300,50*nc);
kh = cfg.kh; %frequency

% Test obstacle Frechet derivative for Dirichlet problem
bc = [];
bc.type = 'Dirichlet';
bc.invtype = 'o';

src0 = [0.01;-0.12];
opts = [];
opts.test_analytic = false;
opts.src_in = src0;
opts.verbose = false;

% set target locations
% receptors (r_{\ell})
r_tgt = cfg.r_tgt;
n_tgt = cfg.n_tgt;

% Incident directions (d_{j})
n_dir = cfg.n_dir;


if strcmp(data_type, 'random') || strcmp(data_type, 'nn')
    rng(pred_idx+ndata)
    coef = sample_fc(cfg, 1);
end

if strcmp(data_type, 'nn') && nc_test > 0
    % generate low freq data, test with high freq predictor
    if nc_test > nc
        error("nc_test must be less than or equal to nc");
    end
    coef = coef .* [ones(1,nc_test+1),zeros(1,nc-nc_test), ones(1,nc_test),zeros(1,nc-nc_test)];
end

if strcmp(data_type, 'nn_stored')
    coef = nn_pred.coef_val(pred_idx, :);
    coef_pred = nn_pred.coef_pred(pred_idx, :);
    src_info_pred = starn(coef_pred,nc,n);
    idx = strfind(pred_path, '_');
    model_name = pred_path(idx(end)+1:end-4);
    idx = strfind(pred_path, '/');
    model_path = [pred_path(1:idx(end)) model_name];
end

src_info_ex = starn(coef,nc,n);
uscat = compute_field(coef,nc,n,kh,n_dir,n_tgt,r_tgt);

if strcmp(data_type, 'nn')
    % apply the stored predictor
    temp_pred_path = strcat(dirname, '/temp.mat');
    coefs_all = coef;
    noise = randn(n_dir,n_tgt) * noise_level;
    uscat_all = reshape(uscat + noise, [1,n_dir, n_tgt]);
    save(temp_pred_path, 'coefs_all', 'uscat_all', 'cfg_str');
    [status,cmdout] = system(strcat(env_path, ' predict.py --data_path=', temp_pred_path,...
        ' --model_path=', model_path, ' --print_coef=True'));
    k = strfind(cmdout,'start to print the coefficients'); % this string has length 31
    coef_pred = str2num(cmdout(k+32:end))';
    src_info_pred = starn(coef_pred,nc,n);
end

umeas = uscat;
model_name = [model_name '_star'];
inverse_inputs = [];
inverse_inputs.nc = nc;
inverse_inputs.kh = kh;
inverse_inputs.n_dir = n_dir;
inverse_inputs.n_tgt = n_tgt;
inverse_inputs.n = n;
inverse_inputs.r_tgt = r_tgt;

if test_origin_alg
    coef_out = starn_specific_inverse(umeas, [1,zeros(1,2*nc)],inverse_inputs);
    src_info_default_res = starn(coef_out,nc,n);
end
figure
hold on
plot(src_info_ex.xs,src_info_ex.ys,'k.', 'MarkerSize', 12);
if test_origin_alg
    plot(src_info_default_res.xs,src_info_default_res.ys,'b--', 'LineWidth',2);
end

if strcmp(data_type, 'random')
    plot(0, 0, 'r*');
    if test_origin_alg
        legend('true boundary', 'boundary solved from default init', '')
    else
        legend('true boundary', '')
    end
elseif strcmp(data_type, 'nn_stored') || strcmp(data_type, 'nn')
    coef_out = starn_specific_inverse(umeas, coef_pred,inverse_inputs);
    src_info_pred_res = starn(coef_out,nc,n);
    
    inverse_result = [src_info_pred.xs; src_info_pred.ys; src_info_pred_res.xs; src_info_pred_res.ys;...
        src_info_ex.xs; src_info_ex.ys]; %pred; refined; true
    d1 = pdist2(inverse_result(1:2,:)', inverse_result(5:6,:)');
    d2 = pdist2(inverse_result(3:4,:)', inverse_result(5:6,:)');
    err_Chamfer = [mean([min(d1), min(d1,[],2)']), mean([min(d2), min(d2,[],2)'])]; %pred, refined
    fprintf('Chamfer difference for DL prediction: %0.3e, for DL refined: %0.3e\n', err_Chamfer(1), err_Chamfer(2))
    err_l2 = [norm(coef - coef_pred) / norm(coef), norm(coef - coef_out) / norm(coef)];
    fprintf('L2 difference for DL prediction: %0.3e, for DL refined: %0.3e\n', err_l2(1), err_l2(2))
    tmp_dir = fullfile(model_path, 'inverse');
    if ~exist(tmp_dir, 'dir')
        mkdir(tmp_dir)
    end
    save(fullfile(tmp_dir, ['inverse' num2str(pred_idx) '.mat']), "inverse_result", "err_Chamfer", "err_l2");
    plot(src_info_pred.xs,src_info_pred.ys,'r:', 'LineWidth',2);
    plot(src_info_pred_res.xs,src_info_pred_res.ys,'m-.', 'LineWidth',2);
    plot(0, 0, 'r*');
    if test_origin_alg
        legend('true boundary', 'boundary solved from default init', 'boundary predicted by nn', 'boundary solved from pred init', '')
    else
        legend('true boundary', 'boundary predicted by nn', 'boundary solved from pred init', '')
    end
w = 9;
h = 8;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [w h]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 w h]);
set(gcf, 'renderer', 'painters');
tmp_dir = fullfile(model_path, 'figs');
if ~exist(tmp_dir, 'dir')
    mkdir(tmp_dir)
end
fig_path = fullfile(tmp_dir, ['nc' int2str(nc) '_k' int2str(kh) '_' model_name '_' int2str(pred_idx) '.pdf']);
print(gcf, '-dpdf', fig_path);
end
end