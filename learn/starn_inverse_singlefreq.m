% This script solves inverse problem based on single frequency data. The
% data is either generated randomly in this script or read from pred.mat

close all
clearvars

data_type = 'nn'; % 'random' or 'nn';
pred_path = './data/star3_kh10_100/test_pred.mat';
nn_pred = load(pred_path);
if strcmp(data_type, 'nn')
    cfg_str = nn_pred.cfg_str;
elseif strcmp(data_type, 'random')
    cfg_path = './configs/nc3.json';
    cfg_str = fileread(cfg_path);
end
cfg = jsondecode(cfg_str);

n  = 300;

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


if strcmp(data_type, 'random')
    coef = sample_fc(cfg, 1);
elseif strcmp(data_type, 'nn')
    pred_idx = 1;
    coef = nn_pred.coef_val(pred_idx, :);
    coef_pred = nn_pred.coef_pred(pred_idx, :);
    src_info_pred = geometries.starn(coef_pred,nc,n);
end

nppw = 20;


src_info_ex = geometries.starn(coef,nc,n);
L = src_info_ex.L;
for ik=1:nk
   n = ceil(nppw*L*abs(kh(ik))/2/pi);
   n = max(n,300);
   src_info_ex = geometries.starn(coef,nc,n);

   [mats,erra] = rla.get_fw_mats(kh(ik),src_info_ex,bc,sensor_info,opts);
   fields = rla.compute_fields(kh(ik),src_info_ex,mats,sensor_info,bc,opts);
end


u_meas = cell(1,1);
u_meas0 = [];
u_meas0.kh = kh;
u_meas0.uscat_tgt = fields.uscat_tgt;
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
%optim_opts.eps_res = 1e-10;
%optim_opts.eps_upd = 1e-10;
opts.store_src_info = true;
[inv_data_all,src_info_out] = rla.rla_inverse_solver(u_meas,bc,...
                          optim_opts,opts);
iter_count = inv_data_all{1}.iter_count;
src_info_default_res = inv_data_all{1}.src_info_all{iter_count};

figure
hold on
plot(src_info_ex.xs,src_info_ex.ys,'k.', 'MarkerSize', 12);
plot(src_info_default_res.xs,src_info_default_res.ys,'b--', 'LineWidth',2);
plot(0, 0, 'r*');

if strcmp(data_type, 'random')
    legend('true boundary', 'boundary solved by default init')
elseif strcmp(data_type, 'nn')
    [inv_data_all_pred,src_info_out_pred] = rla.rla_inverse_solver(u_meas,bc,...
                          optim_opts,opts,src_info_pred);
    iter_count = inv_data_all_pred{1}.iter_count;
    src_info_pred_res = inv_data_all_pred{1}.src_info_all{iter_count};
    plot(src_info_pred.xs,src_info_pred.ys,'r:', 'LineWidth',2);
    plot(src_info_pred_res.xs,src_info_pred_res.ys,'m-.', 'LineWidth',2);
    legend('true boundary', 'boundary solved by default init', 'boundary predicted by nn', 'boundary solved by pred init')
end

% saveas(gcf, ['./figs/pred' int2str(pred_idx) '.pdf'], 'pdf');

