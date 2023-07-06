function T = dprimelmat(kh,src,t)%N,L)
% function to generate the matrix for the derivative of the double layer potential
% Input:
% N -> number of boundary points, should be even
% L -> arclegnth
% src -> domain
% src(1,N) = x(t)
% src(2,N) = y(t)
% src(3,N) = x'(t)
% src(4,N) = y'(t)
% src(5,N) = x''(t)
% src(6,N) = y''(t)
% Output:
% T is the normal derivative of the double layer potential

xs  = src(1,:);
ys  = src(2,:);
dx  = src(3,:);
dy  = src(4,:);
dx2 = src(5,:);
dy2 = src(6,:);
dx3 = src(7,:);
dy3 = src(8,:);

N  = length(xs);
n  = N/2;
C  = 0.57721566490153286060651209008240243104215933593992; %euler-mascheroni's constant

rr = sqrt(bsxfun(@minus,xs',xs).^2+bsxfun(@minus,ys',ys).^2);
drr = dx.^2+dy.^2;

kernel_m  = 1i / 2 * besselh(0, 1, kh * rr);

kernel_m1 = -1 / ( 2 * pi ) * besselj(0, kh * rr);

kernel_m1_diag = -1 / ( 2 * pi );

kernel_m1(1:N+1:end) = kernel_m1_diag;

kernel_m2 = kernel_m - kernel_m1 .* log(4*sin(bsxfun(@minus,t',t)/2).^2);

kernel_m2_diag = 1i/2 - C/pi - 1/pi *log(kh * sqrt(drr)/2);

kernel_m2(1:N+1:end) = kernel_m2_diag;

n_tilde = (bsxfun(@minus,xs',xs) .* repmat(dx,N,1) + bsxfun(@minus,ys',ys) .* repmat(dy,N,1) ) .* ...
    (bsxfun(@minus,xs',xs) .* repmat(dx',1,N) + bsxfun(@minus,ys',ys) .* repmat(dy',1,N) ) ./ rr.^2;

kernel_n  = 1i/2 * n_tilde .* (kh^2 * besselh(0,1,kh*rr) - 2 * kh * besselh(1,1,kh*rr)./rr) + ...
    1i/2 * kh * ( bsxfun(@times,dx',dx)  + bsxfun(@times,dy',dy) ) .* besselh(1,1,kh*rr) ./ rr + ...
    1 ./ (4*pi*sin(bsxfun(@minus,t',t)/2).^2); 

kernel_n1 = -1/(2*pi) * n_tilde .* (kh^2 * besselj(0,kh*rr) - 2 * kh * besselj(1,kh*rr) ./rr ) - ...
    kh/(2*pi) * ( bsxfun(@times,dx',dx)  + bsxfun(@times,dy',dy) ) .* besselj(1,kh*rr) ./ rr;

kernel_n1_diag = -kh^2/(4*pi) * drr; 

kernel_n1(1:N+1:end) = kernel_n1_diag;

kernel_n2 = kernel_n - kernel_n1 .* log(4*sin(bsxfun(@minus,t',t)/2).^2);
    
kernel_n2_diag = ( pi * 1i - 1 - 2 * C - 2 * log( kh * sqrt(drr) / 2 ) ) * kh^2 .* drr / ( 4 * pi )  + ...
    1/( 12 * pi ) + ( dx .* dx2 + dy .* dy2 ).^2 ./ ( 2 * pi * drr.^2 ) - ...
    ( dx2 .* dx2 + dy2 .* dy2 ) ./ ( 4 * pi * drr ) - ...
    ( dx .* dx3 + dy .* dy3 ) ./ ( 6 * pi * drr );

kernel_n2(1:N+1:end) = kernel_n2_diag;

kernel_k1 = kh^2 * kernel_m1 .* ( bsxfun(@times,dx',dx)  + bsxfun(@times,dy',dy) ) - kernel_n1 ;

kernel_k2 = kh^2 * kernel_m2 .* ( bsxfun(@times,dx',dx)  + bsxfun(@times,dy',dy) ) - kernel_n2 ;

quad_r = layer_quad(n);

quad_t = derlayer_quad(n);

T = 0.5 ./ repmat(sqrt(drr)',1,N) .* (gallery('circul',quad_r) .* kernel_k1 + ...
    pi/n * kernel_k2 + gallery('circul',quad_t));

% S1= 0.5 ./ repmat(sqrt(drr)',1,N) .* (gallery('circul',quad_r).* kernel_m1 +...
%     pi/n * kernel_m2) .* ( bsxfun(@times,dx',dx)  + bsxfun(@times,dy',dy) );

