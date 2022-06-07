function [coefs] = sample_fc(cfg, varargin)
    if nargin == 2
        ndata = varargin{1};
    else
        ndata = cfg.ndata;
    end
    nc = cfg.nc;
    coefs = rand(ndata, 2*nc+1);
    coefs(:, 1) = coefs(:, 1) * (cfg.fc_cst_range(2) - cfg.fc_cst_range(1)) + cfg.fc_cst_range(1);
    % each Fourier coefficent is i.i.d. sampled from a uniform distribution
    if strcmp(cfg.dist_type, 'uniform_square')
        coefs(:, 2:2*nc+1) = coefs(:, 2:2*nc+1) * 2*cfg.fc_max - cfg.fc_max;
    % each pair of Fourier coefficients are jointly sampled, whose radius
    % is unformly sampled
    elseif strcmp(cfg.dist_type, 'uniform_radius')
        r = rand(ndata, nc) * cfg.fc_max ./ (1:nc).^(0);
        theta = rand(ndata, nc) * 2 * pi;
        coefs(:, 2:(nc+1)) = r .* cos(theta);
        coefs(:, (nc+2):end) = r.* sin(theta);
    else
        error('Unknown distribution to sample fourier coefficients');
    end
end