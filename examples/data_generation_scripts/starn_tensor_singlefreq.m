% This script generates the data for a star shaped domain, for a fixed
% number of sensors and incident directions where data is available for all
% sensors at each incident direction

clear all 
close all

addpath('../');

n  = 300;
ndata = 1;
% max number of wiggles
nc = 3;

% Set of frequencies (k_{i})
nk = 1;
kh = 10;


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
r_tgt = 10;
n_tgt = 48;
t_tgt = 0:2*pi/n_tgt:2*pi-2*pi/n_tgt;

% Incident directions (d_{j})
n_dir = 48;
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
coefs_all = rand(ndata, 2*nc+1);
coefs_all(:, 1) = coefs_all(:, 1) * 0.2 + 1;
coefs_all(:, 2:2*nc+1) = coefs_all(:, 2:2*nc+1) * 0.6 - 0.3;

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
plot(src_info.xs,src_info.ys,'b.');

% fname = ['../data/star' int2str(nc) '_' int2str(kh) '_' int2str(ndata) '.mat'];
% save(fname, 'coefs_all', 'uscat_all');
