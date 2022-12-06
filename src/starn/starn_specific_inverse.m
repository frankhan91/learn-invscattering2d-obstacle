function coef_out = starn_specific_inverse(umeas, coef_init, other_inputs)
% This script compute the star-shaped inverse obstacle scattering problem
% with single frequency given the far field scattered wave and a initial prediction
% of Fourier coefficients of the radius. In the previous code tgt is at mesh
% grid while here at single grid. The umeas is transposed in this code.

umeas = transpose(umeas);
kh = other_inputs.kh;
var_bd = coef_init;
N_var = other_inputs.nc;
n_bd = other_inputs.n;
n_dir = other_inputs.n_dir;
n_tgt = other_inputs.n_tgt;
if other_inputs.partial
    t_dir = 0:2*pi/n_dir:pi-2*pi/n_dir; %t_dir = t_dir + pi;
    t_tgt = 0:2*pi/n_tgt:pi-2*pi/n_tgt; %t_tgt = t_tgt + pi;
    n_dir = n_dir/2; n_tgt= n_tgt/2;
else
    t_dir = 0:2*pi/n_dir:2*pi-2*pi/n_dir;
    t_tgt = 0:2*pi/n_tgt:2*pi-2*pi/n_tgt;
end
x_dir = cos(t_dir);
y_dir = sin(t_dir);
x_t   = other_inputs.r_tgt * cos(t_tgt);
y_t   = other_inputs.r_tgt * sin(t_tgt); 
tgt   = [ x_t; y_t];
iffilter = 0; 
sigma = 0.1;

%newton variables
flag_newton = 1;
it_newton   = 1;
if isfield(other_inputs, 'eps_step')
    eps_step = other_inputs.eps_step;
else
    eps_step    = 5e-8;
end
if isfield(other_inputs, 'eps_res')
    eps_res = other_inputs.eps_res;
else
    eps_res     = 1e-6;
end
if isfield(other_inputs, 'max_it')
    max_it = other_inputs.max_it;
else
    max_it      = 20;
end
rhs_old     = 1e16;
if isfield(other_inputs, 'alpha') %size for step in the newton method
    alpha = other_inputs.alpha;
else
    alpha = 1;
end


while flag_newton
    
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
    rhs= umeas(1:n_tgt, 1:n_dir)-uscat;
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
    
    %update domain
    var_bd_temp = var_bd + alpha * delta;
    
    %check if there is self intersection; if yes, use Gaussian filter
    num_pts = 20 * N_var;
    ts = 0: (2*pi/num_pts) : (2*pi - 2*pi/num_pts);
    radius= (cos(ts'*(0:N_var))*var_bd_temp(1:(N_var+1))' + sin(ts'*(1:N_var))*var_bd_temp((N_var+2):end)').';
    for idx=1:10
        if prod(radius>0) ==1
            var_bd = var_bd_temp;
            if idx > 1
                fprintf("Self intersection resolved through Gaussian filter")
            end
            break
        else
            sigma0 = 10^(1-idx);
            delta = delta .* exp( - [0:N_var, 1:N_var].*2 / (sigma0^2 * N_var^2));
            var_bd_temp = var_bd + alpha * delta;
            radius = (cos(ts'*(0:N_var))*var_bd_temp(1:(N_var+1))' + sin(ts'*(1:N_var))*var_bd_temp((N_var+2):end)').';
        end
    end
    if prod(radius>0) ==0
        fprintf("Self intersection cannot be resolved, stop iteration")
        break
    end

    %stopping parameter
    if norm(delta)/norm(var_bd) < eps_step
        flag_newton = 0;
        fprintf('Step too small!\n')            
    end
    
    if it_newton >= max_it
        flag_newton = 0;
        fprintf('Reached max iteration!\n')            
    end
    
    if norm(rhs(:))/norm(umeas(:)) < eps_res
        flag_newton = 0;
        fprintf('RHS too small!\n')
    end
    
    if norm(rhs_old)<norm(rhs)
        flag_newton = 0;
        var_bd = var_bd - alpha * delta;
        fprintf('RHS increasing!\n')
    end
    rhs_old = rhs;
    
    fprintf('Iteration =%d\n',it_newton)
    fprintf('RHS =%d\n',norm(rhs(:))/norm(umeas(:)))
    fprintf('Step =%d\n',norm(delta)/norm(var_bd))
    
    it_newton = it_newton + 1;
end
coef_out = var_bd;
% refined: var_bd; init: coef_init; true: coef
% init_error = norm(coef_init - coef) / norm(coef)
% refined_error = norm(coef_out - coef) / norm(coef)
% src_refined = geometries.starn(var_bd,nc,n);
end