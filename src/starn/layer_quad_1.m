function quad = layer_quad_1(n)
% return quadrature in 2n points

j    = 0:(2*n-1);
m    = 1:n-1;

quad = -2*(pi/n)*sum(cos((pi/n)*bsxfun(@times,j',m))./m,2)' + (-1).^j*pi/(n^2);