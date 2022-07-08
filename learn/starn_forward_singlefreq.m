% This script generates the data for a star shaped domain, for a fixed
% number of sensors and incident directions where data is available for all
% sensors at each incident direction
function starn_forward_singlefreq(array_id)
close all
clearvars -except array_id
tic
if nargin == 0
    fprintf('Runing the code on one server, no parallel computing \n')
elseif array_id == 0
    fprintf('Generating validation data \n')
else
    fprintf(['Generating training data through parallel computing with array index ' ...
    num2str(array_id) ', will not generate validation data. \n'])
end

cfg_path = './configs/nc20.json';
data_prefix = '';
cfg_str = fileread(cfg_path);
cfg = jsondecode(cfg_str);
n  = 300;

ndata = cfg.ndata;
nvalid = cfg.nvalid;
% max number of wiggles
nc = cfg.nc;
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
rng(ndata+nvalid)
coefs_val = sample_fc(cfg, nvalid);

nppw = max(2*nc, 20);
coefs = coefs_val(1, :)';
src_info = geometries.starn(coefs,nc,n);
L = src_info.L;
n = max(300, ceil(nppw*L*abs(kh)/2/pi));

dirname = ['./data/star' int2str(nc) '_kh' int2str(kh) '_n' int2str(n_tgt) '_' int2str(ndata)];
if ~strcmp(data_prefix, '')
    dirname = strcat(dirname, '_', data_prefix);
end
if ndata>1 && ~exist(dirname, 'dir')
    mkdir(dirname)
end
train_data_dir = strcat(dirname, '/train_data');
if ndata>1 && ~exist(train_data_dir, 'dir')
    mkdir(train_data_dir)
end

if nargin == 0 || array_id == 0
    uscat_val = zeros(nvalid, n_dir, n_tgt);
    parfor idx=1:nvalid
        coefs = coefs_val(idx, :)';
        src_info = geometries.starn(coefs,nc,n);
        [mats,~] = rla.get_fw_mats(kh,src_info,bc,sensor_info,opts);
        fields = rla.compute_fields(kh,src_info,mats,sensor_info,bc,opts);
        uscat_val(idx, :, :) = reshape(fields.uscat_tgt, [n_dir, n_tgt]);
    end

    figure
    uscat_tgt = squeeze(uscat_val(1, :, :));
    % imagesc(abs(fftshift(fft2(uscat_tgt))))
    imagesc(abs(((uscat_tgt))))
    figure
    hold on
    plot(src_info.xs,src_info.ys,'b.');
    plot(0, 0, 'r*');

    if ndata>1 
        fname = strcat(dirname, '/valid_data.mat');
        save(fname, 'coefs_val', 'uscat_val', 'cfg_str');
        fprintf('Successfully saved the validation data \n')
    end
end
save_fcn = @(name, coefs, uscat) save(name, 'coefs', 'uscat');
if nargin == 0
    fprintf('Start to generate training data \n')
    parfor idx=1:ndata
        coefs = sample_fc(cfg, 1)';
        src_info = geometries.starn(coefs,nc,n);
        [mats,~] = rla.get_fw_mats(kh,src_info,bc,sensor_info,opts);
        fields = rla.compute_fields(kh,src_info,mats,sensor_info,bc,opts);
        uscat = reshape(fields.uscat_tgt, [n_dir, n_tgt]);
        data_name = strcat(train_data_dir, '/train_data_', num2str(idx),'.mat');
        save_fcn(data_name, coefs, uscat);
    end
elseif array_id >= 1
    ndata_per_array = cfg.ndata_per_array;
    start_idx = (array_id - 1) * ndata_per_array + 1;
    end_idx = array_id * ndata_per_array;
    end_idx = min(end_idx, ndata);
    fprintf(['Start to generate training data indexed from ' num2str(start_idx) ...
    ' to ' num2str(end_idx) '\n'])
    parfor idx=start_idx:end_idx
        rng(idx)
        coefs = sample_fc(cfg, 1)';
        src_info = geometries.starn(coefs,nc,n);
        [mats,~] = rla.get_fw_mats(kh,src_info,bc,sensor_info,opts);
        fields = rla.compute_fields(kh,src_info,mats,sensor_info,bc,opts);
        uscat = reshape(fields.uscat_tgt, [n_dir, n_tgt]);
        data_name = strcat(train_data_dir, '/train_data_', num2str(idx),'.mat');
        save_fcn(data_name, coefs, uscat);
    end
end
time=toc
end