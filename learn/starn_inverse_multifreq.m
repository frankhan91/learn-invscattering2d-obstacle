% This script solves inverse problem based on multiple frequency data. The
% data is either generated randomly in this script.

close all
clearvars

cfg_path = './configs/nc10.json';
cfg = jsondecode(fileread(cfg_path));
n  = 300;

ndata = 1;

% max number of wiggles
nc = cfg.nc;
% Set of frequencies (k_{i})
dk = 1;
kh = 1:dk:cfg.kh;


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
coefs = sample_fc(cfg, 1);

nk = size(kh, 2);
u_meas = cell(nk,1);
src_info = geometries.starn(coefs,nc,300);
L = src_info.L;


nppw = 20;

for ik=1:nk
   n = ceil(nppw*L*abs(kh(ik))/2/pi);
   n = max(n,300);
   src_info_ex = geometries.starn(coefs,nc,n);
   
   [mats,erra] = rla.get_fw_mats(kh(ik),src_info_ex,bc,sensor_info,opts);
   fields = rla.compute_fields(kh(ik),src_info_ex,mats,sensor_info,bc,opts);
   
   u_meas0 = [];
   u_meas0.kh = kh(ik);
   u_meas0.uscat_tgt = fields.uscat_tgt;
   u_meas0.tgt = sensor_info.tgt;
   u_meas0.t_dir = sensor_info.t_dir;
   u_meas0.err_est = erra;
   u_meas{ik} = u_meas0;
end

optim_opts = [];
opts = [];
opts.verbose=true;
bc = [];
bc.type = 'Dirichlet';
bc.invtype = 'o';
optim_opts.optim_type = cfg.optim_type;
% Gauss-Newton is much faster than SD in recursive linearization
optim_opts.optim_type = 'gn';
optim_opts.filter_type = cfg.filter_type;
%optim_opts.eps_res = 1e-10;
%optim_opts.eps_upd = 1e-10;
opts.store_src_info = true;
[inv_data_all,src_info_out] = rla.rla_inverse_solver(u_meas,bc,...
                          optim_opts,opts);
iter_count = inv_data_all{nk}.iter_count;
src_info_default_res = inv_data_all{nk}.src_info_all{iter_count};
figure
hold on
plot(src_info_ex.xs,src_info_ex.ys,'k.', 'MarkerSize', 12);
plot(src_info_default_res.xs,src_info_default_res.ys,'b--', 'LineWidth',2);
legend('true boundary', 'boundary solved by multi-frequency')
plot(0, 0, 'r*');