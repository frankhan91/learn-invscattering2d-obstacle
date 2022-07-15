% This script solves inverse problem based on single frequency data. The
% data is either generated randomly in this script or read from pred.mat
function starn_inverse_singlefreq(array_id)
close all
clearvars -except array_id

data_type = 'nn'; % 'random' or 'nn_stored' or 'nn';
env_path = readlines('env_path.txt');
env_path = env_path(1); % only read the first line
test_origin_alg = false;
if strcmp(data_type, 'nn_stored')
    % CAREFUL: need to enter manually
    pred_path = './data/star3_kh10_n48_100/valid_predby_test.mat'; 
    nn_pred = load(pred_path);
    cfg_str = nn_pred.cfg_str;
elseif strcmp(data_type, 'random')
    % CAREFUL: need to enter manually
    cfg_path = './configs/nc3.json';
    cfg_str = fileread(cfg_path);
elseif strcmp(data_type, 'nn')
    % CAREFUL: need to enter model_path, nc_test, and noise_level manually
    % the model_path should not end with '/'
    model_path = './data/star3_kh10_n48_100/test';
    nc_test = 0; % use nc in cfg_path if nc_test=0
    noise_level = 0;
    cfg_path = strcat(model_path, '/data_config.json');
    cfg_str = fileread(cfg_path);
    idx = strfind(model_path, '/');
    model_name = model_path(idx(end)+1:end);
end
cfg_str = erase(cfg_str, '\n'); % jsondecode cannot read '\n' (in big data)
cfg = jsondecode(cfg_str);
ndata = cfg.ndata;
n  = 300;
n_curv = 30;
% max number of wiggles
nc = cfg.nc;
% Set of frequencies (k_{i})
nk = cfg.nk;
kh = cfg.kh;

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
%receptors (r_{\ell})
r_tgt = cfg.r_tgt;
n_tgt = cfg.n_tgt;
t_tgt = 0:2*pi/n_tgt:2*pi-2*pi/n_tgt;

% Incident directions (d_{j})
n_dir = cfg.n_dir;
t_dir = 0:2*pi/n_dir:2*pi-2*pi/n_dir;

[t_tgt_grid,t_dir_grid] = meshgrid(t_tgt,t_dir);
t_tgt_grid = t_tgt_grid(:);
t_dir_grid = t_dir_grid(:);
xtgt = r_tgt*cos(t_tgt_grid);
ytgt = r_tgt*sin(t_tgt_grid);
tgt   = [ xtgt'; ytgt'];


sensor_info = [];
sensor_info.tgt = tgt;
sensor_info.t_dir = t_dir_grid;

% parameters 'a'

nppw = max(2*nc, 20);
if strcmp(data_type, 'random') || strcmp(data_type, 'nn')
    rng(array_id+ndata)
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
    pred_idx = array_id;
    coef = nn_pred.coef_val(pred_idx, :);
    coef_pred = nn_pred.coef_pred(pred_idx, :);
    src_info_pred = geometries.starn(coef_pred,nc,n);
end

src_info_ex = geometries.starn(coef,nc,n);
L = src_info_ex.L;
for ik=1:nk
   n = 2*ceil(nppw*L*abs(kh(ik))/4/pi);
   n = max(n,300);
   src_info_ex = geometries.starn(coef,nc,n);
   freq = fft(src_info_ex.H);
   freq_tail = freq(n_curv+2:end-n_curv);
   ratio = norm(freq_tail) / norm(freq);
   fprintf('the true ratio is %2.3f \n', ratio);
   [mats,erra] = rla.get_fw_mats(kh(ik),src_info_ex,bc,sensor_info,opts);
   fields = rla.compute_fields(kh(ik),src_info_ex,mats,sensor_info,bc,opts);
end

if strcmp(data_type, 'nn')
    % apply the stored predictor
    pred_idx = array_id;
    dirname = ['./data/star' int2str(nc) '_kh' int2str(kh) '_n' int2str(n_tgt) '_' int2str(ndata)];
    temp_pred_path = strcat(dirname, '/temp.mat');
    coefs_all = coef;
    noise = randn(n_dir*n_tgt, 1) * noise_level;
    uscat_all = reshape(fields.uscat_tgt + noise, [1,n_dir, n_tgt]);
    save(temp_pred_path, 'coefs_all', 'uscat_all', 'cfg_str');
    [status,cmdout] = system(strcat(env_path, ' predict.py --data_path=', temp_pred_path,...
        ' --model_path=', model_path, ' --print_coef=True'));
    k = strfind(cmdout,'start to print the coefficients'); % this string has length 31
    coef_pred = str2num(cmdout(k+32:end))';
    src_info_pred = geometries.starn(coef_pred,nc,n);
end

if strcmp(data_type, 'nn_stored') || strcmp(data_type, 'nn')
    err_pred = norm(coef - coef_pred) / norm(coef)
end
u_meas = cell(1,1);
u_meas0 = [];
u_meas0.kh = kh;
if strcmp(data_type, 'nn')
    u_meas0.uscat_tgt = fields.uscat_tgt + noise;
else
    u_meas0.uscat_tgt = fields.uscat_tgt;
end
u_meas0.tgt = sensor_info.tgt;
u_meas0.t_dir = sensor_info.t_dir;
u_meas0.err_est = erra;
u_meas{1} = u_meas0;
   
   
optim_opts = [];
opts = [];
opts.verbose=true;
bc = [];
bc.type = 'Dirichlet';
bc.invtype = 'o';
optim_opts.optim_type = cfg.optim_type;
optim_opts.filter_type = cfg.filter_type;
optim_opts.n_curv = n_curv;
optim_opts.optim_type = 'sd';
%optim_opts.eps_curv = 0.1;
%optim_opts.eps_res = 1e-10;
%optim_opts.eps_upd = 1e-10;
opts.store_src_info = true;
if test_origin_alg
    [inv_data_all,src_info_out] = rla.rla_inverse_solver(u_meas,bc,...
                              optim_opts,opts);
    iter_count = inv_data_all{1}.iter_count;
    src_info_default_res = inv_data_all{1}.src_info_all{iter_count};
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
        legend('true boundary', 'boundary solved by default init', '')
    else
        legend('true boundary', '')
    end
elseif strcmp(data_type, 'nn_stored') || strcmp(data_type, 'nn')
    [inv_data_all_pred,src_info_out_pred] = rla.rla_inverse_solver(u_meas,bc,...
                          optim_opts,opts,src_info_pred);
    iter_count = inv_data_all_pred{1}.iter_count;
    src_info_pred_res = inv_data_all_pred{1}.src_info_all{iter_count};
    plot(src_info_pred.xs,src_info_pred.ys,'r:', 'LineWidth',2);
    plot(src_info_pred_res.xs,src_info_pred_res.ys,'m-.', 'LineWidth',2);
    plot(0, 0, 'r*');
    if test_origin_alg
        legend('true boundary', 'boundary solved by default init', 'boundary predicted by nn', 'boundary solved by pred init', '')
    else
        legend('true boundary', 'boundary predicted by nn', 'boundary solved by pred init', '')
    end
end
w = 9;
h = 8;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [w h]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 w h]);
set(gcf, 'renderer', 'painters');
print(gcf, '-dpdf', ['./figs/prednc' int2str(nc) '_k' int2str(kh) '_' int2str(pred_idx) '.pdf']);
end