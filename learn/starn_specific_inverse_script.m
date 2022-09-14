%function [coef_out] = starn_inverse(umeas, kh, coef_init)
% This script compute the star-shaped inverse obstacle scattering problem
% given the far field scattered wave and (maybe) a initial prediction of
% Fourier coefficients of the radius. In the previous code tgt is at mesh
% grid while here at single grid. The umeas is transposed in this code.
% slmat_out and dlmat_out are renamed to slmat_out_new and dlmat_out_new
% due to same name issue. alpha, eps_step, and eps_res depend highly on the
% specific problem setting.

% TODO: figure out the options needed. Currently: umeas, coef_init, kh, 
clear
% the following code is just for debug purpose
pred_path = './data/star5_kh10_n48_500/valid_predby_test.mat';
%pred_path = './data/star20_kh30_n100_80000/valid_predby_bestmodel.mat';
nn_pred = load(pred_path);
cfg_str = nn_pred.cfg_str;
cfg = jsondecode(cfg_str);
nc = cfg.nc;
kh = cfg.kh;
n  = max(300, 50*nc);
% Test obstacle Frechet derivative for Dirichlet problem
bc = [];
bc.type = 'Dirichlet';
bc.invtype = 'o';
pred_idx = 1;
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
coef = nn_pred.coef_val(pred_idx, :);
%coef_init = nn_pred.coef_pred(pred_idx, :);
coef_init = [1, zeros(1,2*nc)];
src_info = geometries.starn(coef,nc,n);
src_init = geometries.starn(coef_init,nc,n);
[mats,~] = rla.get_fw_mats(kh,src_info,bc,sensor_info,opts);
fields = rla.compute_fields(kh,src_info,mats,sensor_info,bc,opts);
umeas = reshape(fields.uscat_tgt, [n_dir, n_tgt]);

%umeas = umeas + 0.03*randn(n_dir, n_tgt);
umeas = transpose(umeas);
%% the real code begins (line 248 from driver_inverse_dir)
var_bd = coef_init;
N_var = nc;
x_dir = cos(t_dir);
y_dir = sin(t_dir);
n_bd = n;
x_t   = r_tgt * cos(t_tgt);
y_t   = r_tgt * sin(t_tgt); 
tgt   = [ x_t; y_t];
iffilter = 0; 
sigma = 0.1;

%newton variables
flag_newton = 1;
it_newton   = 1;
eps_step    = 5e-8;
eps_res     = 1e-6;
max_it      = 20;
rhs_old     = 1e16;
alpha = 1; %size for step in the newton method

while flag_newton
    
    %generating the boundary
%     n_bd  = ceil(32*kh);
%     if mod(n_bd,2) 
%         n_bd = n_bd+1;
%     end
%     if (n_bd < 100)
%         n_bd = 100;
%     end
    t_bd  = 0:2*pi/n_bd:2*pi-2*pi/n_bd;
    c_t   = cos(t_bd);
    s_t   = sin(t_bd);%change here
    p_t   = (var_bd(1)+cos(bsxfun(@times,t_bd',1:N_var))*var_bd(2:N_var+1)'+...
        sin(bsxfun(@times,t_bd',1:N_var))*var_bd(N_var+2:end)')';
    dp_t  = (bsxfun(@times,(1:N_var)',-sin(bsxfun(@times,(1:N_var)',t_bd)))'*var_bd(2:N_var+1)' + ...
        bsxfun(@times,(1:N_var)',cos(bsxfun(@times,(1:N_var)',t_bd)))'*var_bd(N_var+2:end)')';
    d2p_t = (bsxfun(@times,((1:N_var).*(1:N_var))',-cos(bsxfun(@times,(1:N_var)',t_bd)))'*var_bd(2:N_var+1)' + ...
        bsxfun(@times,((1:N_var).*(1:N_var))',-sin(bsxfun(@times,(1:N_var)',t_bd)))'*var_bd(N_var+2:end)')';
    d3p_t = (bsxfun(@times,((1:N_var).*(1:N_var).*(1:N_var))',sin(bsxfun(@times,(1:N_var)',t_bd)))'*var_bd(2:N_var+1)' + ...
        bsxfun(@times,((1:N_var).*(1:N_var).*(1:N_var))',-cos(bsxfun(@times,(1:N_var)',t_bd)))'*var_bd(N_var+2:end)')';
    xs    = p_t .* c_t;
    ys    = p_t .* s_t;
    dxs   = dp_t .* c_t + p_t .* (-s_t);
    dys   = dp_t .* s_t + p_t .* c_t;
    d2xs  = d2p_t .* c_t + 2 * dp_t .* (-s_t) + p_t .* (-c_t);
    d2ys  = d2p_t .* s_t + 2 * dp_t .* c_t + p_t .* (-s_t);    
    d3xs  = d3p_t .* c_t + d2p_t .*(-s_t) - 2 * (d2p_t .* s_t + dp_t .* c_t) - (dp_t .*c_t + p_t .*(-s_t));
    d3ys  = d3p_t .* s_t + d2p_t .* c_t + 2 * (d2p_t .* c_t - dp_t .* s_t) - (dp_t .* s_t + p_t .* c_t);
    ds    = sqrt(dxs.^2 + dys.^2);
    H     = ( dxs .* d2ys - dys .* d2xs )  ./ ( ds.^3 );

    %setting up the bdry for the operators
    src = zeros(8,n_bd);
    src(1,:) = xs;
    src(2,:) = ys;
    src(3,:) = dxs;
    src(4,:) = dys;
    src(5,:) = d2xs;
    src(6,:) = d2ys;
    src(7,:) = d3xs;
    src(8,:) = d3ys;       

    %generating operators
    S  = slmat(kh,src,t_bd);
    D  = dlmat(kh,src,t_bd);
    Sp = sprimelmat(kh,src,t_bd);
%         T  = dprimelmat(kh,src,t_bd);%this guy is not really necessary
    
    S_tgt = slmat_out_new(kh,src,tgt);
    D_tgt = dlmat_out_new(kh,src,tgt);    
            
    %bd_data
    uinc  = exp(1i *kh * (bsxfun(@times,xs',x_dir)+bsxfun(@times,ys',y_dir)));
    duinc = 1i* kh * (bsxfun(@times,dys',x_dir)-bsxfun(@times,dxs',y_dir))./repmat(ds',1,n_dir) .* ...
        exp(1i *kh * (bsxfun(@times,xs',x_dir)+bsxfun(@times,ys',y_dir)));
    
    %fw for q to find dudn (looking at equation 5.52 Colton & Kress
    %book (4th edition). Green's identity gives you Sdudn=u^{inc}.
    %Taking the normal derivative you get the equation below.
    eta = kh;
    Fw_mat1 = Sp + eye(n_bd)/2 - 1i * eta * S;
    inv_Fw1 = inv(Fw_mat1);
    bd_data = duinc - 1i * eta * uinc;
    dudn = inv_Fw1 * bd_data;
           
    %Still neeed to find u^scat
    Fw_mat = D+eye(n_bd)/2+1i*eta*S;
    inv_Fw = inv(Fw_mat);
    
    %scattered field
    bd_data = - uinc;
    pot = inv_Fw * bd_data;
    uscat = (D_tgt + 1i * eta * S_tgt) * pot;
    
    %comparing dudn with our calculation - this is just a check of
    %things. 
%         ubd_aux  = (D + eye(n_bd)/2 + 1i* eta *S ) * pot;
%         fprintf('Error u=%d\n',max(max(abs(ubd_aux+uinc))))
%         dubd_aux = (T + 1i * eta * ( Sp - eye(n_bd)/2 ) ) * pot;
%         fprintf('Error dudn=%d\n',max(max(abs(dudn-(dubd_aux+duinc)))))
                    
    %right hand side
    rhs= umeas-uscat;
    rhs = rhs(:);
    
    %constructing matrix for inverse problem        
    DFw_var    = zeros(length(rhs),2*N_var+1);
    for ivar = 1 : (2*N_var+1)
        %basis delta shape
        delta_var  = zeros(1,2*N_var+1);    
        delta_var(ivar) = 1;            
        h_t = (delta_var(1)+cos(bsxfun(@times,t_bd',1:N_var))*delta_var(2:N_var+1)'+...
                sin(bsxfun(@times,t_bd',1:N_var))*delta_var(N_var+2:end)')';
        hxs    = h_t .* c_t;
        hys    = h_t .* s_t;
                        
        h_nu = repmat((( hxs .* dys - hys .* dxs ) ./ ds)', 1, n_dir );

        bd_data_delta = -h_nu .* dudn ;                
        
        %find potential
        pot = inv_Fw * bd_data_delta;
    
        % measure delta
        DFw_col         = (D_tgt + 1i * eta * S_tgt)*pot;
        DFw_var(:,ivar) = DFw_col(:);
        
    end
            
    %finding delta
    delta = [real(DFw_var); imag(DFw_var)] \ [ real(rhs); imag(rhs) ];
            
    delta = delta';
    %filter, probably not need to low frequencies        
    if (iffilter == 1)
        sol_aux   = 0;            
        hg        = 2/(2*N_var);
        tg        = -1:hg:1;
        gauss_val = exp(-tg.*tg/sigma);
        gauss_new = zeros(1,length(gauss_val)); 
        gauss_new(1:N_var+1) = gauss_val(N_var+1:end);
        gauss_new(N_var+2:end) = gauss_val(N_var:-1:1);
        delta = delta.*gauss_new;
    end        
    
    %update domain - > may not need the damping alpha of the Newton
    %step
    var_bd = var_bd + alpha * delta;
            
    %stopping parameter
    if norm(delta)/norm(var_bd) < eps_step
        flag_newton = 0;
        fprintf('Step too small!\n')            
    end
    
    if it_newton > max_it
        flag_newton = 0;
        fprintf('Reached max iteration!\n')            
    end
    
    if norm(rhs(:))/norm(umeas(:)) < eps_res
        flag_newton = 0;
        fprintf('RHS too small!\n')
    end
    
    if norm(rhs_old)<norm(rhs)
        flag_newton = 0;
        %var_bd = var_bd - alpha * delta;
        fprintf('RHS increasing!\n')
    end
    rhs_old = rhs;
    
    fprintf('Iteration =%d\n',it_newton)
    fprintf('RHS =%d\n',norm(rhs(:))/norm(umeas(:)))
    fprintf('Step =%d\n',norm(delta)/norm(var_bd))
    
    it_newton = it_newton + 1;
end

% refined: var_bd; init: coef_init; true: coef
init_error = norm(coef_init - coef) / norm(coef);
refined_error = norm(var_bd - coef) / norm(coef);
src_refined = geometries.starn(var_bd,nc,n);
%%
figure; hold on
plot(src_info.xs,src_info.ys,'b-', 'LineWidth',2);
plot(src_init.xs,src_init.ys,'r:', 'LineWidth',2);
plot(src_refined.xs,src_refined.ys,'m:', 'LineWidth',2);
plot(0, 0, 'r*');
legend('true boundary', 'boundary predicted by nn', 'boundary solved by pred init', '')
Title = ['starnc' num2str(nc) ' k=' num2str(kh) ' alpha=' num2str(alpha, '%.0e')];
title(Title)
w = 9;
h = 8;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [w h]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 w h]);
set(gcf, 'renderer', 'painters');
fig_path = ['./figs/nc' int2str(nc) '_k' int2str(kh) '_' num2str(pred_idx) '.pdf'];
print(gcf, '-dpdf', fig_path);
