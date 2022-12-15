function starn_lsm_fft(lsm_idx)
close all
clearvars -except lsm_idx
if nargin == 0
    lsm_idx=1;
end
partial = false;
noise_level = 0;
cfg_path = './configs/nc5.json';
cfg_str = fileread(cfg_path);
cfg = jsondecode(cfg_str);
star_specific = true;
% max number of wiggles
nc = cfg.nc;
kh = cfg.kh;
n  = max(300, 50*nc);

% parameters 'a'
rng(cfg.ndata+cfg.nvalid) %in order to have the save validation data with DL method
coefs_all = sample_fc(cfg, cfg.nvalid);
coefs = coefs_all(lsm_idx,:); clear coefs_all
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
% Test obstacle Frechet derivative for Dirichlet problem
bc = [];
bc.type = 'Dirichlet';
bc.invtype = 'o';


src0 = [0.01;-0.12];
opts = [];
opts.test_analytic = true;
opts.src_in = src0;
opts.verbose=true;

m=200;
% set target locations
%receptors (r_{\ell})
r_tgt = 10;
n_tgt = cfg.n_tgt;
n_tgt = m;
t_tgt = 0:2*pi/n_tgt:2*pi-2*pi/n_tgt;

% Incident directions (d_{j})
n_dir = cfg.n_dir;
n_dir = m;
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

src_info = geometries.starn(coefs,nc,n);

[mats,erra] = rla.get_fw_mats(kh,src_info,bc,sensor_info,opts);
fields = rla.compute_fields(kh,src_info,mats,sensor_info,bc,opts);

u_meas = [];
u_meas.kh = kh;
if noise_level == 0
    u_meas.uscat_tgt = fields.uscat_tgt;
else
    rng(lsm_idx)
    noise = 1 + noise_level * rand(n_dir*n_tgt, 1) .* exp(2*pi*1i*rand(n_dir*n_tgt, 1));
    u_meas.uscat_tgt = fields.uscat_tgt .* noise;
end
u_meas.tgt = sensor_info.tgt;
u_meas.t_dir = sensor_info.t_dir;
u_meas.err_est = erra;

alpha = 1e-8;

[Ig,xgrid0,ygrid0] = lsm.lsm_tensor(n_tgt,n_dir,u_meas,alpha,partial);
figure; surf(xgrid0,ygrid0,Ig); shading interp; view(2); hold on;  
plot3(src_info.xs,src_info.ys,100*ones(size(src_info.xs)),'k')

value_max = 7;
value_min = 4.6;
step_size = 0.2;
level_values = value_max:-step_size:value_min;
figure;contour(xgrid0, ygrid0, Ig, level_values); hold on; plot(src_info.xs,src_info.ys,'k')
perimeter = 10000;

% decide a threshold and the boundary
count = 0;
for value = level_values
    count = count + 1;
    level_set = contour(xgrid0,ygrid0,Ig,[value, value]);
    len_total = length(level_set(1,:));
    
    % extract different connected parts
    boundary = {};
    bdry_idx = 0;
    point_idx = 1;
    while point_idx < len_total
        part_len = level_set(2, point_idx);
        cur_points = level_set(:, (point_idx+1):(point_idx+part_len));
        if cur_points(:,1) == cur_points(:,end)
            bdry_idx = bdry_idx+1;
            boundary{bdry_idx} = cur_points;
        end
        point_idx = point_idx + part_len + 1;
    end
    num_connected = bdry_idx;
    
    % find the connected part with the longest perimeter
    cur_perimeter = 0;
    for bdry_idx=1:num_connected
        cur_points = boundary{bdry_idx};
        perim = sum(vecnorm(diff(cur_points,1,2), 2,1) );
        if perim > cur_perimeter
            cur_perimeter = perim;
            points = cur_points;
        end
    end

    %decide whether to stop or to update the shape
    if count > 1
        dist = pdist2(points', bdry_points');
        change_Chamfer = mean([min(dist), min(dist,[],2)']);
        change_perimeter = cur_perimeter - perimeter;
        if change_Chamfer >= 0.1 || abs(change_perimeter) > 1
            % break if meet the above criteria, parameter obtained by hand
            value = value + step_size;
            break
        else
            perimeter = cur_perimeter;
            bdry_points = points;
        end
    else
        perimeter = cur_perimeter;
        bdry_points = points;
    end
end


%% find FFT coefficients
[theta,rho] = cart2pol(bdry_points(1,:),bdry_points(2,:));

%make some data
t = theta(1, 2:end);
x = rho(1, 2:end);
nc_lsm = nc;

%file exchange submission
[afit, bfit, yfit] = Fseries(t,x,nc_lsm);

figure
hold on
plot(t,x)
plot(t,yfit,'g')

legend('original data',['Fit with Fseries to ',num2str(nc_lsm),' frequencies'])

%% check if the extracted curve match the true one
figure; plot(src_info.xs,src_info.ys); hold on;
plot(bdry_points(1,:),bdry_points(2,:));
coefs_lsm_fit = zeros([1,2*nc_lsm+1]);
coefs_lsm_fit(1, 1) = afit(1) / 2;
coefs_lsm_fit(1, 2:nc_lsm+1) = afit(2:end)';
coefs_lsm_fit(1, nc_lsm+2:end) = bfit';
if nc == nc_lsm
    err_l2 = norm(coefs - coefs_lsm_fit) / norm(coefs)
else
    err_l2 = -1;
end
src_lsm = geometries.starn(coefs_lsm_fit,nc_lsm,n);
plot(src_lsm.xs,src_lsm.ys);

dist = pdist2([src_info.xs; src_info.ys]', bdry_points');
err_Chamfer1 = mean([min(dist), min(dist,[],2)'])
err_l2_refined = -1;
%% apply the inverse algorithm
if star_specific
    umeas = reshape(u_meas.uscat_tgt, [n_dir, n_tgt]);
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
    coef_out = starn_specific_inverse(umeas, coefs_lsm_fit,inverse_inputs);
    src_info_lsm_res = geometries.starn(coef_out,nc,n);
    err_l2_refined = norm(coefs - coef_out) / norm(coefs)
else
    bc = [];
    bc.type = 'Dirichlet';
    bc.invtype = 'o';
    
    optim_opts = [];
    opts = [];
    opts.verbose=true;
    bc = [];
    bc.type = 'Dirichlet';
    bc.invtype = 'o';
    optim_opts.optim_type = 'sd';
    optim_opts.filter_type = cfg.filter_type;
    opts.store_src_info = true;
    u_meas_all = cell(1,1);
    u_meas_all{1} = u_meas;
    [inv_data_all_lsm,src_info_out_lsm] = rla.rla_inverse_solver(u_meas_all,bc,...
                                optim_opts,opts,src_lsm);
    iter_count = inv_data_all_lsm{1}.iter_count;
    src_info_lsm_res = inv_data_all_lsm{1}.src_info_all{iter_count};
end
%%
plot(src_info_lsm_res.xs,src_info_lsm_res.ys);
legend('true boundary', 'lsm boundary', 'fitted star shape from lsm boundary', 'refined boundary')
dist_refined = pdist2([src_info_lsm_res.xs; src_info_lsm_res.ys]', [src_info.xs; src_info.ys]');
err_Chamfer_refined = mean([min(dist_refined), min(dist_refined,[],2)']);
err_Chamfer = [err_Chamfer1, err_Chamfer_refined];
inverse_result = [src_lsm.xs; src_lsm.ys; src_info_lsm_res.xs; src_info_lsm_res.ys;...
        src_info.xs; src_info.ys]; %lsm; refined; true
if ~exist('./data/lsm', 'dir')
    mkdir('./data/lsm');
end
model_dir = ['./data/lsm/inverse' num2str(nc)];
if noise_level > 0
    model_dir = [model_dir 'n' num2str(10*noise_level)];
end
if partial
    model_dir = [model_dir 'p'];
end
if ~exist(model_dir, 'dir')
    mkdir(model_dir);
    mkdir([model_dir '/inverse']);
end
save([model_dir '/inverse/inverse' num2str(lsm_idx) '.mat'], "inverse_result", "err_Chamfer", "err_l2", "err_l2_refined")
fprintf(['Chamfer error before refine ' num2str(err_Chamfer1) ', after refine ' num2str(err_Chamfer_refined) '\n'])
w = 9;
h = 8;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [w h]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 w h]);
set(gcf, 'renderer', 'painters');
fig_path = ['./figs/nc' int2str(nc) '_k' int2str(kh) '_nclsm' int2str(nc_lsm) '_' int2str(lsm_idx) '.pdf'];
print(gcf, '-dpdf', fig_path);
end