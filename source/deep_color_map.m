function deep_color_map( connected, savefig )
%
% Plot the latent coordinates that points are mapped to by a deep function
% (the coordinates are represented by colors)
%
% David Duvenaud
% Sept 2013

if nargin < 3; seed = 0; end;   % random seed
if nargin < 2; savefig = false; end    % Do we save the images
if nargin < 1; connected = false; end  % Does the input connect to every layer

if connected
    basedir = sprintf('../figures/latent_seed_%d_large_connected/', seed);
else
    basedir = sprintf('../figures/latent_seed_%d_large/', seed);
end

mkdir(basedir);

layers = 50;
n_1d = 20;
n_1d_aux = 500;

addpath(genpath('utils'));

% Fix the seed of the random generators.
randn('state',seed);
rand('state',seed);



xrange = linspace( -1, 1, n_1d);

% Make a grid
[xa, xb] = ndgrid( xrange, xrange ); %randn(n, D);   % Deepest latent layer.
x0 = [ xa(:), xb(:) ];

xrange2 = linspace( -1, 1, n_1d_aux );
[xa, xb] = ndgrid( xrange2, xrange2 ); %randn(n, D);   % Deepest latent layer.
x0aux = [ xa(:), xb(:) ];


hold on;
axis off
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');

x = x0;
xaux = x0aux;


fig_title = 'Base Distribution';
%title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);
colors = coord_to_color(sin(2.*pi.*xaux));
im = reshape(colors, n_1d_aux, n_1d_aux, 3);
imshow(im);
%set_fig_units_cm( 8,8 );
axis tight
if savefig
    imwrite(im, [basedir 'layer_0.png'], 'png' );
    %savepng(gcf, [basedir 'layer_0']);
end



for l = 1:layers
    fprintf('Layer %d\n', l);
    % This mean and variance will be repeated for each dimension.
    mu = zeros(size(x, 1), 1);
    
    % Potentially augment the state with the original inputs.
    if connected
        aug = [x x0];
        augaux = [xaux x0aux];
    else
        aug = x;
        augaux = xaux;
    end
    
    
    % Sampling the warping
    % ==========================
    sigma = se_kernel(aug', aug') + eye(size(aug,1)) * 1e-6;
    %figure(2); imagesc(K)

    k_x_xaux = se_kernel(aug', augaux');

    % Now sample second deepest layer.
    x(:,1) = mvnrnd( mu, sigma);
    x(:,2) = mvnrnd( mu, sigma);

    % Work out warping distribution conditional on the already sampled points.
    xaux = k_x_xaux' / sigma * x;
    %sigmacond = k_x_xaux' / sigma * k_x_xaux;

    % Now sample some more points, conditioned on those.
    %xaux(:,1) = mvnrnd( mucond(:,1), sigmacond);
    %xaux(:,2) = mvnrnd( mucond(:,2), sigmacond);

    % R = RandOrthMat(2);  % Neural network version
    % xaux = 10./(1 + exp(-xaux*R));

    
    
    % Drawing the warping
    % ==========================
    
    colors = coord_to_color(sin(2.*pi.*xaux));
    im = reshape(colors, n_1d_aux, n_1d_aux, 3);
    clf; imshow(im);
    hold on;
    %plot(x0(:,1), x0(:,2), 'k.');
    axis off
    set( gca, 'XTick', [] );
    set( gca, 'yTick', [] );
    set( gca, 'XTickLabel', '' );
    set( gca, 'yTickLabel', '' );
    set(gcf, 'color', 'white');
    set(gca, 'YGrid', 'off');
    
    %border_scale = 10;
    %xlim( good_xlim );
    %ylim( good_ylim );
    
    %set_fig_units_cm( 8,8 );
    %axis tight
    %set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
    %fig_title = sprintf('Layer %d', l);
    %title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);
    if savefig
        imwrite(im, sprintf([basedir 'latent_coord_map_layer_%d.png'], l), 'png' );
        %savepng(gcf, sprintf([basedir 'latent_coord_map_layer_%d'], l));
    end
    drawnow;
end


end

% Kernel function
function sigma = se_kernel(x, y)
    if nargin == 0
        sigma = 'Normal SE covariance.'; return;
    end

    sigma = 0.9*exp( -0.5.*sq_dist(x, y));
end
