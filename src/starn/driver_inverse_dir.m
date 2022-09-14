%driver for inverse dirichlet
clear

%set up bounday
N_bd          = 10;
coefs_bd      = zeros(1,2*N_bd+1);
coefs_bd(1)   = 1;
coefs_bd(N_bd+1) = 0.5;

%incident field frequency
dk    = 0.25;
n_kh  = 30;
khv   = 1:dk:n_kh*dk;

% incidence directions
n_dir = 16;
t_dir = 0:2*pi/n_dir:2*pi-2*pi/n_dir;
x_dir = cos(t_dir);
y_dir = sin(t_dir);
dir =[ x_dir; y_dir ];
    
%receptors
r_tgt = 10;
n_tgt = 100;
t_tgt = 0:2*pi/n_tgt:2*pi-2*pi/n_tgt;
x_t   = r_tgt * cos(t_tgt);
y_t   = r_tgt * sin(t_tgt);    
tgt   = [ x_t; y_t];

%choose to add noise
ifnoise   = 0;
noise_lvl = 0.02;

%for inverse problem parameters go to the inverse problem part of the code



% Carlos' data genration script
for ik = 1 : length(khv)    
    %incident data    
    kh = khv(ik);
    fprintf('Generating data for k=%d\n',kh)
                 
    %generating the boundary
    % you dont need as many points here. It can totally be changed
    %overall the only restriction is that the number of points must be even
    n_bd  = ceil(40*kh);
    if mod(n_bd,2) 
        n_bd = n_bd+1;
    end
    if (n_bd < 500)
        n_bd = 500;
    end
    t_bd  = 0:2*pi/n_bd:2*pi-2*pi/n_bd;
    c_t   = cos(t_bd);
    s_t   = sin(t_bd);
    p_t   = (coefs_bd(1)+cos(bsxfun(@times,t_bd',1:N_bd))*coefs_bd(2:N_bd+1)'+...
        sin(bsxfun(@times,t_bd',1:N_bd))*coefs_bd(N_bd+2:end)')';
    dp_t  = (bsxfun(@times,(1:N_bd)',-sin(bsxfun(@times,(1:N_bd)',t_bd)))'*coefs_bd(2:N_bd+1)' + ...
        bsxfun(@times,(1:N_bd)',cos(bsxfun(@times,(1:N_bd)',t_bd)))'*coefs_bd(N_bd+2:end)')';
    d2p_t = (bsxfun(@times,((1:N_bd).*(1:N_bd))',-cos(bsxfun(@times,(1:N_bd)',t_bd)))'*coefs_bd(2:N_bd+1)' + ...
        bsxfun(@times,((1:N_bd).*(1:N_bd))',-sin(bsxfun(@times,(1:N_bd)',t_bd)))'*coefs_bd(N_bd+2:end)')';
    d3p_t = (bsxfun(@times,((1:N_bd).*(1:N_bd).*(1:N_bd))',sin(bsxfun(@times,(1:N_bd)',t_bd)))'*coefs_bd(2:N_bd+1)' + ...
        bsxfun(@times,((1:N_bd).*(1:N_bd).*(1:N_bd))',-cos(bsxfun(@times,(1:N_bd)',t_bd)))'*coefs_bd(N_bd+2:end)')';
    xs    = p_t .* c_t;
    ys    = p_t .* s_t;
    dxs   = dp_t .* c_t + p_t .* (-s_t);
    dys   = dp_t .* s_t + p_t .* c_t;
    d2xs  = d2p_t .* c_t + 2 * dp_t .* (-s_t) + p_t .* (-c_t);
    d2ys  = d2p_t .* s_t + 2 * dp_t .* c_t + p_t .* (-s_t);    
    d3xs  = d3p_t .* c_t + d2p_t .*(-s_t) - 2 * (d2p_t .* s_t + dp_t .* c_t) - (dp_t .*c_t + p_t .*(-s_t));
    d3ys  = d3p_t .* s_t + d2p_t .* c_t + 2 * (d2p_t .* c_t - dp_t .* s_t) - (dp_t .* s_t + p_t .* c_t);
    ds    = sqrt(dxs.^2 + dys.^2);
                
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
    
    %bd_data
    uinc  = exp(1i *kh * (bsxfun(@times,xs',x_dir)+bsxfun(@times,ys',y_dir)));
    bd_data = -uinc;
    
    %calculating the measurements
    %generating operators
    S  = slmat(kh,src,t_bd);
    D  = dlmat(kh,src,t_bd);
    Sp = sprimelmat(kh,src,t_bd);    
    
    %solving the system
    eta    = kh;    
    Fw_mat = D+eye(n_bd)/2+1i*eta*S;
    inv_Fw = inv(Fw_mat);
    pot    = inv_Fw * bd_data;

    %calculating scattered field at target
    S_tgt = slmat_out(kh,src,tgt);
    D_tgt = dlmat_out(kh,src,tgt);
    umeas(ik).data = (D_tgt + 1i * eta * S_tgt)*pot;
    
    %add noise
    if (ifnoise == 1)
        noise = randn(n_tgt,n_dir)+1i*randn(n_tgt,n_dir);
        umeas(ik).data = umeas(ik).data + noise_lvl ...
            * abs(umeas(ik).data).* noise./abs(noise);
     end
    
end
clear umeas

% converter for inverse_obstacle scattering script to Carlos format

A = load('~/git/inverse-obstacle-scattering2d/examples/data/star10_ik1_nk27_tensor_data_Dirichletcarlos_test.mat');
nk = length(khv);
for i=1:nk
   umeas(i).data = reshape(A.u_meas{i}.uscat_tgt,[n_dir,n_tgt]).';
end

% start inverse problem here

%filter parameters
iffilter = 0; 
sigma = 0.1;


%Time to solve the inverse problem
%RLA with Newton method
for ik = 1 : length(khv)
    
    %incident data
    kh = khv(ik);
    
    fprintf('Wavenumber=%d\n',kh)
    
    if ik == 1
        % set initial guess for domain, by default set to unit circle       
        N_var      = floor(2*kh);
        var_bd    = zeros(1,2*N_var+1);
        var_bd(1) = 1;        
    else
        N_var_old    = N_var;
        var_bd_old  = var_bd;        
        N_var        = floor(2*kh);
        var_bd      = zeros(1,2*N_var+1);
        var_bd(1)   = var_bd_old(1);
        var_bd( 2 : N_var_old + 1 ) = var_bd_old( 2 : N_var_old + 1 );
        var_bd( N_var + 2 : N_var + N_var_old + 1 ) = var_bd_old( N_var_old +2 : end );
    end
    
    %newton variables
    flag_newton = 1;
    it_newton   = 1;
    eps_step    = 1e-3;
    eps_res     = 1e-2;
    max_it      = 100;
    rhs_old     = 1e16;
    alpha = 0.1; %size for step in the newton method
    
    while flag_newton
        
        %generating the boundary
        n_bd  = ceil(32*kh);
        if mod(n_bd,2) 
            n_bd = n_bd+1;
        end
        if (n_bd < 100)
            n_bd = 100;
        end
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
        
        S_tgt = slmat_out(kh,src,tgt);
        D_tgt = dlmat_out(kh,src,tgt);    
                
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
        rhs= umeas(ik).data-uscat;
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
        
        if norm(rhs(:))/norm(umeas(ik).data(:)) < eps_res
            flag_newton = 0;
            fprintf('RHS too small!\n')
        end
        
        if norm(rhs_old)<norm(rhs)
            flag_newton = 0;
            var_bd = var_bd - delta;
            fprintf('RHS increasing!\n')
        end
        rhs_old = rhs;
        
        fprintf('Iteration =%d\n',it_newton)
        fprintf('RHS =%d\n',norm(rhs(:))/norm(umeas(ik).data(:)))
        fprintf('Step =%d\n',norm(delta)/norm(var_bd))
        
        it_newton = it_newton + 1;
    end
    
    bd_vecs(ik).coefs = var_bd;
    
end


%To see the video keep pressing enter
figure; hold on;
h1 = plot(0,0);
for ik=1:length(khv)
    n_bd  = ceil(32*kh);
    if mod(n_bd,2) 
        n_bd = n_bd+1;
    end
    if (n_bd < 64)
        n_bd = 64;
    end
    t_bd   = 0:2*pi/n_bd:2*pi-2*pi/n_bd;
    c_t    = cos(t_bd);
    s_t    = sin(t_bd);
    %old domain
    p_t    = (coefs_bd(1)+cos(bsxfun(@times,t_bd',1:N_bd))*coefs_bd(2:N_bd+1)'+...
        sin(bsxfun(@times,t_bd',1:N_bd))*coefs_bd(N_bd+2:end)')';
    xs     = p_t .* c_t;
    ys     = p_t .* s_t;
        
    %new domain
    var_bd = bd_vecs(ik).coefs;
    N_var  = (length(var_bd)-1)/2;
    bd_t   = (var_bd(1)+cos(bsxfun(@times,t_bd',1:N_var))*var_bd(2:N_var+1)'+...
            sin(bsxfun(@times,t_bd',1:N_var))*var_bd(N_var+2:end)')';
    xs1    = bd_t .* c_t;
    ys1    = bd_t .* s_t;
        
    h0 = plot([xs xs(1)], [ys ys(1)],'b');
    delete(h1)        
    h1 = plot([xs1 xs(1)], [ys1 ys(1)],'-.k');
    pause;
    
end

