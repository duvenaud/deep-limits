function deep_gp_sample_1d
%
% Plot the latent coordinates that points are mapped to by a deep function
%
% David Duvenaud
% Sept 2013

% Options:
connected = true;  % Does the input connect to every layer
seed=1;
layers = 100;
sample_resolution = 400;
n_1d = 15000;
savefig = false;



addpath(genpath('utils'));

% Fix the seed of the random generators.
randn('state',seed);
rand('state',seed);

if connected
    basedir = sprintf('../figures/latent_seed_%d_1d_large_connected/', seed);
else
    basedir = sprintf('../figures/latent_seed_%d_1d_large/', seed);
end
mkdir(basedir);


x0 = linspace( -4, 4 , n_1d )';
x = x0;

for l = 1:layers
    
    % Optionally augment current inputs with the original x.
    if connected
        augx = [x x0];
    else
        augx = x;
    end
    
    % Sample a random function in the domain of the current inputs.
    lower_domain = min(augx);
    upper_domain = max(augx);

    randfunc = sample_function( lower_domain, upper_domain, sample_resolution, @se_kernel);

    y = randfunc(augx);
    
    
       
    % Plot the composed function.
    clf;
    subplot(2,2,1); 
    plot( x0, y, 'b- '); hold on;
    fig_title = sprintf('Layer %d Compostion', l);
    title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);    
    
    if ~connected
    cur_xrange = linspace(lower_domain, upper_domain, sample_resolution)';
    % Plot the current layer's function.
    subplot(2,2,2); 
    plot( cur_xrange, randfunc(cur_xrange), 'b- '); hold on;
    % Show where the data coming is was.
    plot( augx, zeros(size(augx)), 'k.' );
    fig_title = sprintf('Layer %d Function', l);
    title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);    
    
    
    subplot(2,2,3); 
    % Show where the data coming is was.
    hist( augx, 100 );
    fig_title = sprintf('Layer %d Input density', l);
    title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);    
    end
    
    set(gcf, 'color', 'white');
    

    pause;
    
    x = y; 
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
    sigma = kernel(x', x');
    sigma = sigma + eye(size(x,1)) * 1e-4 * max(sigma(:));
    mu = zeros(size(x, 1), 1);
    y = mvnrnd( mu, sigma);
    
    % Random function is the conditional mean.
    randfunc = @(xp) kernel(x', xp')' / sigma * y';
end

function sigma = se_kernel(x, y)
    if nargin == 0
        sigma = 'Normal SE covariance.'; return;
    end

    ell = sqrt(2/pi);
    sigma_output = 1;
    
    (sigma_output^2/ell^2) * 2 / pi
    
    %sigma = pi/2 .* exp( -0.5.*sq_dist(x, y)) .*10;
    sigma = sigma_output^2 .* exp( -0.5.*sq_dist(x, y) ./ ell^2) ;
end
