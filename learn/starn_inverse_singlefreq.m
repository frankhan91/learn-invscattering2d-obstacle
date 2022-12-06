% This script solves inverse problem based on single frequency data. The
% data is either generated randomly in this script or read from pred.mat
function starn_inverse_singlefreq(pred_idx)
close all
clearvars -except pred_idx

data_type = 'nn_stored'; % 'random' or 'nn_stored' or 'nn';
star_specific = true;
env_path = readlines('env_path.txt');
env_path = env_path(1); % only read the first line
test_origin_alg = false;
partial = false;
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
n_curv = 100;
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

if strcmp(data_type, 'random') || strcmp(data_type, 'nn')
    rng(pred_idx+ndata)
    coef = sample_fc(cfg, 1);
    % coefs for the figures
    % nc=5, k=5
    %coef = [1.00647342205048	0.00869796052575111	-0.0303471498191357
    % 0.000751996121834964	0.0479891784489155	-0.0489483885467052
    % -0.0330795720219612	-0.0122160883620381	-0.0566566064953804
    % -0.0533075556159020	0.0667197480797768];

    % nc=10, k=10
    %coef = [1.06346285343170	-0.0202399902045727	-0.0170682966709137
    % -0.0356857813894749	-0.00907227210700512	-0.0243198294192553
    % -0.00444695958867669	0.00213610520586371	0.0144090102985501
    % -0.0701758041977882	0.00326031912118197	-0.0140952924266458
    % 0.00273356749676168	0.0793705955147743	-0.0348840728402138
    % -0.00611411174759269	0.00945145450532436	0.0149966273456812
    % -0.0208816770464182	0.0593406111001968	0.0543491393327713];
    
    % nc=20, k=30
    %coef = [1.13417851924896	0.0319831594824791	0.00969072338193655
    % 0.0138962203636765	0.0147123169153929	-0.0254861395806074
    % 0.0354111380875111	-0.0244231950491667	0.00566629925742745	
    % 0.00692472420632839	-0.0374359302222729	-0.0273015685379505
    % 0.0703499987721443	-0.0704057291150093	-0.0162532664835453
    % -0.00317577272653580	0.0641473233699799	-0.0606653206050396	
    % -0.0153619600459933	-0.00540020968765020	-0.0123984999954700
    % -0.0314480774104595	0.0397983677685261	-0.0481126308441162	
    % 0.0346182174980640	0.0797742232680321	-0.0384230874478817
    % -0.0228993073105812	0.0375300496816635	0.0232317112386227
    % 0.0200495086610317	0.0325807854533196	-0.0137067576870322
    % 0.0246882103383541	0.0174419805407524	0.0942149385809898
    % 0.0158416125923395	-0.0291298329830170	0.00178801163565367
    % -0.00437810551375151	0.0968383029103279];
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
    src_info_pred = geometries.starn(coef_pred,nc,n);
    idx = strfind(pred_path, '_');
    model_name = pred_path(idx(end)+1:end-4);
    idx = strfind(pred_path, '/');
    model_path = [pred_path(1:idx(end)) model_name];
end

src_info_ex = geometries.starn(coef,nc,n);
freq = fft(src_info_ex.H);
freq_tail = freq(n_curv+2:end-n_curv);
ratio = norm(freq_tail) / norm(freq);
fprintf('the true ratio is %2.3f \n', ratio);
[mats,erra] = rla.get_fw_mats(kh,src_info_ex,bc,sensor_info,opts);
fields = rla.compute_fields(kh,src_info_ex,mats,sensor_info,bc,opts);

if strcmp(data_type, 'nn')
    % apply the stored predictor
    dirname = ['./data/star' int2str(nc) '_kh' int2str(kh) '_n' int2str(n_tgt) '_' int2str(ndata)];
    temp_pred_path = strcat(dirname, '/temp.mat');
    coefs_all = coef;
    rng(0)
    noise = 1 + noise_level * rand(n_dir*n_tgt, 1) .* exp(2*pi*1i*rand(n_dir*n_tgt, 1));
    uscat_all = reshape(fields.uscat_tgt .* noise, [1,n_dir, n_tgt]);
    save(temp_pred_path, 'coefs_all', 'uscat_all', 'cfg_str');
    [status,cmdout] = system(strcat(env_path, ' predict.py --data_path=', temp_pred_path,...
        ' --model_path=', model_path, ' --print_coef=True'));
    k = strfind(cmdout,'start to print the coefficients'); % this string has length 31
    coef_pred = str2num(cmdout(k+32:end))';
    src_info_pred = geometries.starn(coef_pred,nc,n);
end

if strcmp(data_type, 'nn_stored') || strcmp(data_type, 'nn')
    err_l2 = norm(coef - coef_pred) / norm(coef)
end
u_meas = cell(1,1);
u_meas0 = [];
u_meas0.kh = kh;
if strcmp(data_type, 'nn')
    u_meas0.uscat_tgt = fields.uscat_tgt .* noise;
else
    u_meas0.uscat_tgt = fields.uscat_tgt;
end
u_meas0.tgt = sensor_info.tgt;
u_meas0.t_dir = sensor_info.t_dir;
u_meas0.err_est = erra;
u_meas{1} = u_meas0;

if star_specific
    umeas = reshape(u_meas0.uscat_tgt, [n_dir, n_tgt]);
    model_name = [model_name '_star'];
    inverse_inputs = [];
    inverse_inputs.nc = nc;
    inverse_inputs.kh = kh;
    inverse_inputs.n_dir = n_dir;
    inverse_inputs.n_tgt = n_tgt;
    inverse_inputs.n = n;
    inverse_inputs.r_tgt = r_tgt;
    inverse_inputs.partial = partial;
%     inverse_inputs.alpha = 1;
%     inverse_inputs.eps_step = 5e-8;
%     inverse_inputs.eps_res = 1e-6;
%     inverse_inputs.max_it      = 20;
else
    optim_opts = [];
    opts = [];
    opts.verbose=true;
    bc = [];
    bc.type = 'Dirichlet';
    bc.invtype = 'o';
    optim_opts.optim_type = cfg.optim_type;
    optim_opts.filter_type = cfg.filter_type;
    optim_opts.n_curv = n_curv;
    optim_opts.optim_type = 'sd'; model_name=[model_name optim_opts.optim_type];
    %optim_opts.eps_curv = 0.1;
    %optim_opts.eps_res = 1e-10;
    %optim_opts.eps_upd = 1e-10;
    opts.store_src_info = true;
end

if test_origin_alg
    if star_specific
        coef_out = starn_specific_inverse(umeas, [1,zeros(1,2*nc)],inverse_inputs);
        src_info_default_res = geometries.starn(coef_out,nc,n);
        err_l2_refined_orig = norm(coef - coef_out) / norm(coef)
    else
        [inv_data_all,src_info_out] = rla.rla_inverse_solver(u_meas,bc,...
                                    optim_opts,opts);
        iter_count = inv_data_all{1}.iter_count;
        src_info_default_res = inv_data_all{1}.src_info_all{iter_count};
    end
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
    if star_specific
        coef_out = starn_specific_inverse(umeas, coef_pred,inverse_inputs);
        src_info_pred_res = geometries.starn(coef_out,nc,n);
        err_l2_refined = norm(coef - coef_out) / norm(coef)
    else
        [inv_data_all_pred,~] = rla.rla_inverse_solver(u_meas,bc,...
                                optim_opts,opts,src_info_pred);
        iter_count = inv_data_all_pred{1}.iter_count;
        src_info_pred_res = inv_data_all_pred{1}.src_info_all{iter_count};
    end
    inverse_result = [src_info_pred.xs; src_info_pred.ys; src_info_pred_res.xs; src_info_pred_res.ys;...
        src_info_ex.xs; src_info_ex.ys]; %pred; refined; true
    d1 = pdist2(inverse_result(1:2,:)', inverse_result(5:6,:)');
    d2 = pdist2(inverse_result(3:4,:)', inverse_result(5:6,:)');
    err_Chamfer = [mean([min(d1), min(d1,[],2)']), mean([min(d2), min(d2,[],2)'])] %pred, refined
    if star_specific
        save([model_path '/inverse/inverse' num2str(pred_idx) '.mat'], "coef", "coef_pred", "inverse_result", "err_Chamfer", "err_l2", "err_l2_refined")
    else
        save([model_path '/inverse/inverse' num2str(pred_idx) '.mat'], "coef", "coef_pred", "inverse_result", "err_Chamfer", "err_l2")
    end
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
fig_path = [model_path '/figs/nc' int2str(nc) '_k' int2str(kh) '_' model_name '_' int2str(pred_idx) '.pdf'];
print(gcf, '-dpdf', fig_path);
end