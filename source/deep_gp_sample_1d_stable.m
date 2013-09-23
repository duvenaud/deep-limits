function deep_gp_sample_1d
%
% Plot the latent coordinates that points are mapped to by a deep function
%
% David Duvenaud
% Sept 2013

connected = false;  % Does the input connect to every layer
seed=0;
% Fix the seed of the random generators.
randn('state',seed);
rand('state',seed);

if connected
    basedir = sprintf('../figures/latent_seed_%d_1d_large_connected/', seed);
else
    basedir = sprintf('../figures/latent_seed_%d_1d_large/', seed);
end

mkdir(basedir);

layers = 100;
n_1d = 400;
n_1d_aux = 1500;
savefig = false;

addpath(genpath('utils'));

%x0 = linspace( -10, 1, n_1d)';
x0aux = linspace( -2, 2 , n_1d_aux )';


hold on;
axis off
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');


xaux = x0aux;



for l = 1:layers
    
    if connected
        augaux = [xaux x0aux];
    else
        augaux = xaux;
    end
    
    lower = min(augaux);
    upper = max(augaux);
    resolution = 600;
    randfunc = sample_function( lower, upper, resolution, @se_kernel);

    new_xaux = randfunc(augaux);  
    
    %clf; plot( xaux, new_xaux, 'b- '); hold on;

    xaux = new_xaux; 
    
    clf; plot( x0aux, xaux, 'b- '); hold on;
    
    fig_title = sprintf('Layer %d', l);
    title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);
    pause;
end


end


function randfunc = sample_function( lower, upper, resolution, kernel)
% Generate a random function, of any number of input dimensions,
% by interpolating between at least resolution number of points
% of a draw from a GP prior.
%
% David Duvenaud
% Sept 2013

    D = numel(lower);
    N_1D = ceil(resolution^(1/D));
    N = N_1D^D;
    
    % Create grid.
    xrange = cell(1,D);
    for d = 1:D
        xrange{d} = linspace( lower(d), upper(d), N_1D)';
    end
    grid = cell(1,D);
    [grid{1:D}]=ndgrid(xrange{:});
    
    % Unroll each grid
    for d = 1:D
        temp = grid{d};
        temp = temp(:);
        grid{d} = temp;
    end
    x = cell2mat(grid);
    
    % Sample function at grid points.
    sigma = kernel(x', x') + eye(size(x,1)) * 1e-6;
    mu = zeros(size(x, 1), 1);
    y = mvnrnd( mu, sigma);
    
    % Random function is the conditional mean.
    randfunc = @(xp) kernel(x', xp')' / sigma * y';
end

function sigma = se_kernel(x, y)
    if nargin == 0
        sigma = 'Normal SE covariance.'; return;
    end

    %sigma = pi/2 .* exp( -0.5.*sq_dist(x, y)) .*10;
    sigma = 1/2 .* exp( -0.5.*sq_dist(x, y) .* 10) ;
end
