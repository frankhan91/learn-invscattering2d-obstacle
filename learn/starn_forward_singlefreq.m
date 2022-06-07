% This script generates the data for a star shaped domain, for a fixed
% number of sensors and incident directions where data is available for all
% sensors at each incident direction

close all
clearvars

cfg_path = './configs/nc3.json';
data_prefix = '';
cfg_str = fileread(cfg_path);
cfg = jsondecode(cfg_str);
n  = 300;

ndata = cfg.ndata;

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
coefs_all = sample_fc(cfg);

nppw = 20;

uscat_all = zeros(ndata, n_dir, n_tgt);
for idx=1:ndata
    coefs = coefs_all(idx, :)';
    src_info = geometries.starn(coefs,nc,n);
    L = src_info.L;
    for ik=1:nk
       n = ceil(nppw*L*abs(kh(ik))/2/pi);
       n = max(n,300);
       src_info = geometries.starn(coefs,nc,n);

       [mats,erra] = rla.get_fw_mats(kh(ik),src_info,bc,sensor_info,opts);
       fields = rla.compute_fields(kh(ik),src_info,mats,sensor_info,bc,opts);
    end
    uscat_all(idx, :, :) = reshape(fields.uscat_tgt, [n_dir, n_tgt]);
end

figure
uscat_tgt = squeeze(uscat_all(ndata, :, :));
% imagesc(abs(fftshift(fft2(uscat_tgt))))
imagesc(abs(((uscat_tgt))))

figure
hold on
plot(src_info.xs,src_info.ys,'b.');
plot(0, 0, 'r*');

dirname = ['./data/star' int2str(nc) '_kh' int2str(kh) '_' int2str(ndata)];
if ~strcmp(data_prefix, '')
    dirname = strcat(dirname, '_', data_prefix);
end
if ~exist(dirname, 'dir')
   mkdir(dirname)
end
fname = strcat(dirname, '/forward_data.mat');
save(fname, 'coefs_all', 'uscat_all', 'cfg_str');
