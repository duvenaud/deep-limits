function deep_gp_connected
%
% Plot sampled distributions from a deep GP model.
%
% David Duvenaud
% Dec 2102


% Settings
seed=0;
savefig = false;
n = 400;
n_aux = 25000;
D = 2;
layers = 10;



addpath('utils');

% Fix the seed of the random generators.
randn('state',seed);
rand('state',seed);

basedir = sprintf('../figures/deep_gp_sample_seed_%d/', seed);
mkdir(basedir);


% Sample deepest latent density.
x0 = randn(n, D) ./ 1.5;
x0aux = randn(n_aux, D);
x = x0; xaux = x0aux;

% Plot base density.
circle_colors = coord_to_color(xaux);
plot_density( x, xaux, 1, circle_colors, 'Original latent space' );
if savefig
    savepng(gcf, sprintf([basedir 'deep_sample_layer0'], layer));
end

for layer = 2:layers
    % Specify the mean and covariance of the warping.
    % This mean and variance will be repeated for each dimension.
    mu = zeros(1, n);
    sigma = se_kernel([x x0]', [x x0]') + eye(n) * 1e-6;
    k_x_xaux = se_kernel([x x0]', [xaux x0aux]');

    % Now sample next deepest layer.
    x(:,1) = mvnrnd( mu, sigma);
    x(:,2) = mvnrnd( mu, sigma);

    % Work out warping distribution,
    % conditional on the already sampled points.
    mucond = k_x_xaux' / sigma * x;
    %sigmacond = k_x_xaux' / sigma * k_x_xaux;

    % Now sample some more points, conditioned on those.
    %xaux(:,1) = mvnrnd( mucond(:,1), sigmacond);
    %xaux(:,2) = mvnrnd( mucond(:,2), sigmacond);
    xaux(:,1) = mucond(:,1);
    xaux(:,2) = mucond(:,2);
    
    % Plot the warped points and auxiliary points.
    clf;
    plot_density( x, xaux, circle_colors, basedir );
    savepng(gcf, sprintf([basedir 'deep_sample_connected_layer%d'], layer));
end


end


function sigma = se_kernel(x, y)
    if nargin == 0
        sigma = 'Normal SE covariance.'; return;
    end

    sigma = exp( -0.5.*sq_dist(x, y));
end

function plot_density( x, xaux, circle_colors)
    
    circle_size = .005;
    circle_alpha = 0.16;
    
    h_dots = plot(x(:,1), x(:,2), 'w.');
    good_xlim = xlim;
    good_ylim = ylim;
    
    plot_little_circles(xaux(:,1), xaux(:,2), ...
         circle_size, circle_colors, circle_alpha, false ); hold on;    
    hold on;
    set(h_dots, 'visible', 'off');
    
    axis off
    set( gca, 'XTick', [] );
    set( gca, 'yTick', [] );
    set( gca, 'XTickLabel', '' );
    set( gca, 'yTickLabel', '' );
    set(gcf, 'color', 'white');
    set(gca, 'YGrid', 'off');
    
    xlim( good_xlim );
    ylim( good_ylim );
    
    set_fig_units_cm( 14, 14 );
    set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
    title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);
end
