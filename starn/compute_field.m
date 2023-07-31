function uscat = compute_field(coef,nc,n_bd,kh,n_dir,n_tgt,r_tgt)
% parameters needed
var_bd = coef;
N_var = nc;
t_dir = 0:2*pi/n_dir:2*pi-2*pi/n_dir;
t_tgt = 0:2*pi/n_tgt:2*pi-2*pi/n_tgt;
x_dir = cos(t_dir);
y_dir = sin(t_dir);
x_t   = r_tgt * cos(t_tgt);
y_t   = r_tgt * sin(t_tgt); 
tgt   = [ x_t; y_t];

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
%ds    = sqrt(dxs.^2 + dys.^2);
%H     = ( dxs .* d2ys - dys .* d2xs )  ./ ( ds.^3 );

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
%Sp = sprimelmat(kh,src,t_bd);
%         T  = dprimelmat(kh,src,t_bd);%this guy is not really necessary

S_tgt = slmat_out_new(kh,src,tgt);
D_tgt = dlmat_out_new(kh,src,tgt);    
        
%bd_data
uinc  = exp(1i *kh * (bsxfun(@times,xs',x_dir)+bsxfun(@times,ys',y_dir)));
% duinc = 1i* kh * (bsxfun(@times,dys',x_dir)-bsxfun(@times,dxs',y_dir))./repmat(ds',1,n_dir) .* ...
%     exp(1i *kh * (bsxfun(@times,xs',x_dir)+bsxfun(@times,ys',y_dir)));

%fw for q to find dudn (looking at equation 5.52 Colton & Kress
%book (4th edition). Green's identity gives you Sdudn=u^{inc}.
%Taking the normal derivative you get the equation below.
eta = kh;
%Fw_mat1 = Sp + eye(n_bd)/2 - 1i * eta * S;
%inv_Fw1 = inv(Fw_mat1);
%bd_data = duinc - 1i * eta * uinc;
%dudn = inv_Fw1 * bd_data;
       
%Still neeed to find u^scat
Fw_mat = D+eye(n_bd)/2+1i*eta*S;
inv_Fw = inv(Fw_mat);

%scattered field
bd_data = - uinc;
pot = inv_Fw * bd_data;
uscat = (D_tgt + 1i * eta * S_tgt) * pot;
uscat = transpose(uscat);
end