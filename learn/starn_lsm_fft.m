close all
clearvars
cfg_path = './configs/nc5.json';
cfg_str = fileread(cfg_path);
cfg = jsondecode(cfg_str);
% max number of wiggles
nc = cfg.nc;
kh = cfg.kh;
kh=5;
n  = max(300, 50*nc);

% parameters 'a'
rng(0)
coefs = sample_fc(cfg, 1);
% coefs = [1.1065    0.0303   -0.0099    0.0208   -0.1834    0.1053   -0.0658   -0.0723    0.0413    0.2207    0.2602];
% coefs = [1.1731   -0.1557   -0.0933   -0.0094    0.1546    0.0249   -0.1576   -0.1490   -0.0734   -0.1267   -0.0028];
% coefs = [1.0908   -0.1437    0.1244    0.1961    0.0937    0.0688    0.0658    0.0121   -0.0197    0.1634    0.0542];
% coefs = [1.0258    0.0518    0.1438    0.0072    0.1657    0.2268    0.0564    0.2133   -0.0158   -0.1273   -0.0258];
% coefs = [1.1980   -0.1341   -0.0365    0.0409   -0.0296   -0.0360    0.0110    0.0952   -0.0225   -0.0836    0.2121];
% coefs = [1.0118   -0.0632    0.2416   -0.0661    0.1001   -0.2760   -0.2827    0.0132    0.2706   -0.1546    0.1030];
% coefs = [1.0295   -0.1482    0.0032   -0.2533   -0.0440    0.1414   -0.0604   -0.0031   -0.1481   -0.1662    0.1909];

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
u_meas.uscat_tgt = fields.uscat_tgt;
u_meas.tgt = sensor_info.tgt;
u_meas.t_dir = sensor_info.t_dir;
u_meas.err_est = erra;

alpha = 1e-8;

[Ig,xgrid0,ygrid0] = lsm.lsm_tensor(n_tgt,n_dir,u_meas,alpha);
figure; surf(xgrid0,ygrid0,Ig); shading interp; view(2); hold on;  
plot3(src_info.xs,src_info.ys,100*ones(size(src_info.xs)),'k')

value_max = 7;
value_min = 4.6;
step_size = 0.2;
level_values = value_max:-step_size:value_min;
figure;M=contour(xgrid0, ygrid0, Ig, level_values); hold on; plot(src_info.xs,src_info.ys,'k')
perimeter = 10000;

%% decide a threshold and the boundary
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
        len_part = length(cur_points(1,:));
        perim = sum(vecnorm(diff(cur_points,1,2), 2,1) );
        if perim > cur_perimeter
            cur_perimeter = perim;
            points = cur_points;
            longest_index = bdry_idx;
            len = len_part;
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
nc_lsm = 20;

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
src_lsm = geometries.starn(coefs_lsm_fit,nc_lsm,n);
plot(src_lsm.xs,src_lsm.ys);
legend('true boundary', 'lsm boundary', 'fitted star shape from lsm boundary')
dist = pdist2([src_info.xs; src_info.ys]', bdry_points');
err_Chamfer = mean([min(dist), min(dist,[],2)'])