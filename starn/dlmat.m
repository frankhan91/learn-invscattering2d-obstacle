function D = dlmat(kh,src,t)%N,L)
% function to generate the matrix for the double layer potential
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
% D is the double layer potential


xs  = src(1,:);
ys  = src(2,:);
dx  = src(3,:);
dy  = src(4,:);
dx2 = src(5,:);
dy2 = src(6,:);

N  = length(xs);
n  = N/2;

rr = sqrt(bsxfun(@minus,xs',xs).^2+bsxfun(@minus,ys',ys).^2);
drr = dx.^2+dy.^2;

kernel = (1i*kh/4)*(bsxfun(@minus,xs',xs).*repmat(dy,N,1)-bsxfun(@minus,ys',ys).*repmat(dx,N,1)).* ...
    besselh(1,1,kh*rr)./rr;

kernel_1 = -kh/(4*pi)*(bsxfun(@minus,xs',xs).*repmat(dy,N,1)-bsxfun(@minus,ys',ys).*repmat(dx,N,1)).* ...
    besselj(1,kh*rr)./rr;

kernel_1(1:N+1:end) = zeros(N,1);
    
kernel_2 = kernel - kernel_1.*log(4*sin(bsxfun(@minus,t',t)/2).^2);

kernel_2_diag = -1/(4*pi)*(dx.*dy2-dy.*dx2)./drr;

kernel_2(1:N+1:end) = kernel_2_diag;

quad = layer_quad(n);

D = (gallery('circul',quad).*kernel_1 + pi/n*kernel_2);