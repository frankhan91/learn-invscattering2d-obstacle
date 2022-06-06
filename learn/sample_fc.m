function [coefs] = sample_fc(cfg, varargin)
    if nargin == 2
        ndata = varargin{1};
    else
        ndata = cfg.ndata;
    end
    nc = cfg.nc;
    coefs = rand(ndata, 2*nc+1);
    coefs(:, 1) = coefs(:, 1) * (cfg.fc_cst_range(2) - cfg.fc_cst_range(1)) + cfg.fc_cst_range(1);
    if strcmp(cfg.dist_type, 'uniform_square')
        coefs(:, 2:2*nc+1) = coefs(:, 2:2*nc+1) * 2*cfg.fc_max - cfg.fc_max;
    else
        error('Unknown distribution to sample fourier coefficients');
    end
end