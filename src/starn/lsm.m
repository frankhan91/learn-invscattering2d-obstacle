clear
%info about incoming waves
%k -> wavenumber
%Nd -> number of directions
kh = 1;
Nd = 100;
hd = 2*pi/Nd;
t_dir = 0:hd:2*pi-hd;
x_dir = cos(t_dir);
y_dir = sin(t_dir);

% sensor locations
radius = 10;
tgt = radius * [cos(t_dir);sin(t_dir)];

%domain info
% dflag : 1 -> circle
%         2 -> kite
%         3 -> ellipse 2:1
%         4 -> trig polynomial times (cos(t),sin(t))
% The following only matters for dflag == 4.
% N_bd -> number coeffs of trig polynomial (total: 2 * N_bd + 1)
% coefs_bd -> coeffs trig polynomial
Nt = 128;
ht = 2*pi/Nt;
t = 0:ht:2*pi;
t = t(1:end-1);
dflag = 4;
N_bd  = 5;
coefs_bd = zeros(1,2 * N_bd + 1);
coefs_bd(1) = 1;
coefs_bd(N_bd+1) = 0.3;
if dflag == 1
    %circle
    xs  = cos(t);
    ys  = sin(t);
    dxs = - sin(t);
    dys = cos(t);
    d2xs = - cos(t);
    d2ys = - sin(t);
    
elseif dflag == 2
    %kite
    xs = cos(t) + 0.65 * cos(2 * t) - 0.65;
    ys = 1.5 * sin(t);
    dxs = - sin(t) - 1.3 * sin(2 * t);
    dys = 1.5 * cos(t);
    d2xs = - cos(t) - 2.6 * cos(2 * t);
    d2ys = - 1.5 * sin(t);

elseif dflag == 3
    % ellipse 2:1
    a = 2.0;
    b = 1.0;
    c_t = cos(t);
    s_t = sin(t);
    p_t = b./sqrt((b/a)^2*cos(t).^2+sin(t).^2);
    dp_t = -b/2*((1-(b/a)^2)*sin(2*t))./sqrt((b/a)^2*cos(t).^2+sin(t).^2).^1.5;    
    d2p_t = b *((3 * (sin(2*t) - (b^2 * sin(2*t))/a^2).^2)./ ...
        (4 *((b^2 * cos(t).^2)/a^2 + sin(t).^2).^(5/2)) - ...
        ((2 * b^2 * sin(t).^2)/a^2 - (2 * b^2 * cos(t).^2)/a^2 - ...
        2 * sin(t).^2 + 2 * cos(t).^2)./(2 *((b^2* cos(t).^2)/a^2 + ...
        sin(t).^2).^(3/2)));
    xs   = p_t .* c_t;
    ys   = p_t .* s_t;
    dxs  = dp_t .* c_t + p_t .* (-s_t);
    dys  = dp_t .* s_t + p_t .* c_t;
    d2xs = d2p_t .* c_t + 2 * dp_t .* (-s_t) + p_t .* (-c_t);
    d2ys = d2p_t .* s_t + 2 * dp_t .* c_t + p_t .* (-s_t);    
    
elseif dflag == 4
    % trig polynomial times (cos(t),sin(t))
    c_t   = cos(t);
    s_t   = sin(t);
    p_t   = (coefs_bd(1)+cos(bsxfun(@times,t',1:N_bd))*coefs_bd(2:N_bd+1)'+...
        sin(bsxfun(@times,t',1:N_bd))*coefs_bd(N_bd+2:end)')';
    dp_t  = (bsxfun(@times,(1:N_bd)',-sin(bsxfun(@times,(1:N_bd)',t)))'*coefs_bd(2:N_bd+1)' + ...
        bsxfun(@times,(1:N_bd)',cos(bsxfun(@times,(1:N_bd)',t)))'*coefs_bd(N_bd+2:end)')';
    d2p_t = (bsxfun(@times,((1:N_bd).*(1:N_bd))',-cos(bsxfun(@times,(1:N_bd)',t)))'*coefs_bd(2:N_bd+1)' + ...
        bsxfun(@times,((1:N_bd).*(1:N_bd))',-sin(bsxfun(@times,(1:N_bd)',t)))'*coefs_bd(N_bd+2:end)')';
    xs    = p_t .* c_t;
    ys    = p_t .* s_t;
    dxs   = dp_t .* c_t + p_t .* (-s_t);
    dys   = dp_t .* s_t + p_t .* c_t;
    d2xs  = d2p_t .* c_t + 2 * dp_t .* (-s_t) + p_t .* (-c_t);
    d2ys  = d2p_t .* s_t + 2 * dp_t .* c_t + p_t .* (-s_t);    
    
else
    fprintf('Not implemented!\n')
    return
end
src = zeros(6,Nt);
src(1,:) = xs;
src(2,:) = ys;
src(3,:) = dxs;
src(4,:) = dys;
src(5,:) = d2xs;
src(6,:) = d2ys;

% noise-data
ifnoise = 0;
noise_lvl = 0.01;

% regularization parameter for inverse problem
alpha = 1e-6;

%grid for inverse problem
Ngrid  = 200;
tgrid0 = -3;
tgrid1 = 3;

% forward problem data generation
uinc  = exp(1i *kh * (bsxfun(@times,xs',x_dir)+bsxfun(@times,ys',y_dir)));
S  = slmat(kh,src,t);
D  = dlmat(kh,src,t);
Fw = (eye(Nt)/2 + D + 1i * kh * S);
pot = Fw \ (- uinc);
S_tgt = slmat_out(kh,src,tgt);
D_tgt = dlmat_out(kh,src,tgt);
uscat = (D_tgt + 1i * kh * S_tgt) * pot;

% adding noise
if (ifnoise == 1)
    noise = randn(size(uscat))+1i*randn(size(uscat));
    uscat = uscat + noise_lvl ...
        * abs(uscat).* noise./abs(noise);
end

% LSM -> it is really only this.
% The LSM originally uses the far-field pattern
% The equation is derived in the paper: history
% The far field equation is the equation 2.8 in the paper above, this equation 
% is derived from equation 2.3 in the same paper.
% To use scattered data, I used equation 2.3 in the paper.
% I don't know if the proofs would all work in this case, I would need to
% read over and check the paper, but so far it seems fine in the examples
% here.

% setting up grid for inverse problem
hgrid = (tgrid1 -tgrid0)/(Ngrid-1);
tgrid = tgrid0:hgrid:tgrid1;
[xgrid0,ygrid0] = meshgrid(tgrid);
xgrid = xgrid0(:);
ygrid = ygrid0(:);

% setting up rhs for LSM
b = exp(1i*pi/4)*sqrt(pi*kh/2)*besselh(0,1,kh*sqrt( ...
                              bsxfun(@minus,xgrid,tgt(1,:)).^2 + ...
                              bsxfun(@minus,ygrid,tgt(2,:)).^2));
b = transpose(b);

% setting-up operator
% alpha -> regularization parameter (you can play around, 10^-6 does the
% trick here for the kite
%
A = uscat*hd;
F = A' * A + alpha * eye(Nd);
fprintf('Condition number of LSM operator:%d\n',cond(A))
fprintf('Condition number of LSM problem after regularization:%d\n',cond(F))

% Solving for LSM
% For each point you will have a vector. g is a matrix Nd times Ngrid
g = F \ (A' * b);

% Calculating the indicator function at each point in the grid. 
% Just like the paper above we have
% Ig(z) = log(\|g(z)\|_L^2). 
% I didn't add the hd here, it is unecessary.
% Other indicator functions can be used. A search in the literature must be
% done to find a more suitable one.
Ig = reshape(log(sqrt(sum(conj(g).*g,1))),Ngrid,Ngrid);
figure; surf(xgrid0,ygrid0,Ig); shading interp; view(2); hold on;  
plot3(xs,ys,100*ones(size(xs)),'k')

figure; contour(xgrid0,ygrid0,Ig); hold on; plot(xs,ys,'k')

% At this point, the user should look for a value that would give him the
% contour desired. In a conversation with Peter Monk, he told me that the
% ideal is to do the following:
% a) solve the problem for the circle of radius 1.
% b) find the contour value that better serves the approximation of the
% circle.
% c) use the value obtained in b to search for the countour in IG. This gi
% ves you the domain. 
% I didn't implement this here, but it is not difficult to do.

