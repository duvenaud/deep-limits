function deep_color_map_video( connected, savefig, seed, neural_net )
%
% Plot the latent coordinates that points are mapped to by a deep function
% (the coordinates are represented by colors)
%
% Losslessly made into a movie with:
% ./ffmpeg -i seed_0_large_connected/frame_%d.png -c:v libx264 -qp 0 output3.mkv
%
% David Duvenaud
% Feb 2014

if nargin < 1; connected = true; end  % Does the input connect to every layer
if nargin < 2; savefig = false; end    % Do we save the images
if nargin < 3; seed = 0; end;   % random seed
if nargin < 4; neural_net = false; end   % Use a neural net or a GP warping?


if savefig
    if connected
        basedir = sprintf('/scratch/frames/seed_%d_large_connected/', seed);
    else
        basedir = sprintf('/scratch/frames/seed_%d_large/', seed);
    end
    mkdir(basedir);
end


frames_per_layer = 24;

layers = 50;
n_1d = 20;
n_1d_aux_x = 2000;
n_1d_aux_y = 2000;

minibatch_size = 10000;

num_neurons = 1000;
input_scale = 2;
output_scale = 1;

addpath(genpath('utils'));

% Fix the seed of the random generators.
randn('state',seed);
rand('state',seed);



xrange = linspace( -1, 1, n_1d);

% Make a grid
[xa, xb] = ndgrid( xrange, xrange ); %randn(n, D);   % Deepest latent layer.
x0 = [ xa(:), xb(:) ];

% Should be roughly in range -1 to 1
auxrangex = linspace( -1, 1, n_1d_aux_x ) .* n_1d_aux_x / min(n_1d_aux_x, n_1d_aux_y);
auxrangey = linspace( -1, 1, n_1d_aux_y ) .* n_1d_aux_y / min(n_1d_aux_x, n_1d_aux_y);
[xa, xb] = ndgrid( auxrangex, auxrangey ); %randn(n, D);   % Deepest latent layer.
x0aux = [ xa(:), xb(:) ];
npixels = size(x0aux,1);

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


%fig_title = 'Base Distribution';
%title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);
colors = coord_to_color2(sin(2.*pi.*xaux));
im = reshape(colors, n_1d_aux_x, n_1d_aux_y, 3);
%imshow(im);
%set_fig_units_cm( 8,8 );
axis tight
if savefig
    imwrite(im, [basedir 'layer_0.png'], 'png' );
    %savepng(gcf, [basedir 'layer_0']);
end


frame = 0;
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
    
    xaux_old = xaux;
    
    % Sampling the warping
    % ==========================
    if neural_net
        % Finite neural network version.
        %R = RandOrthMat(size(augaux,2));

        R = randn(size(augaux,2), num_neurons).*input_scale;
        hidden_units = output_scale./(1 + exp(-augaux*R));
        xaux = hidden_units * randn(num_neurons, 2);
    else
        % GP warping
        sigma = se_kernel(aug', aug') + eye(size(aug,1)) * 1e-6;

        % Now sample next deepest layer.
        x(:,1) = mvnrnd( mu, sigma);
        x(:,2) = mvnrnd( mu, sigma);

        % Work out warping distribution conditional on the already sampled points.
        xaux = NaN(npixels, 2);
        for i = 1:minibatch_size:npixels
            cur_ix = i:min(i+minibatch_size-1, npixels);
            k_x_xaux = se_kernel(aug', augaux(cur_ix, :)');
            xaux(cur_ix, :) = k_x_xaux' * (sigma \ x);
        end
        %sigmacond = k_x_xaux' / sigma * k_x_xaux;

        % Now sample some more points, conditioned on those.
        %xaux(:,1) = mvnrnd( mucond(:,1), sigmacond);
        %xaux(:,2) = mvnrnd( mucond(:,2), sigmacond);
    end

    
    for inner = linspace(0, 1, frames_per_layer)
        
        a = sin(inner * pi/2).^ 2;
        b = cos(inner * pi/2).^ 2;
        
        xaux_interp = xaux_old .* b + xaux .* a;
        
        % Drawing the warping
        % ==========================

        colors = coord_to_color2(sin(2.*pi.*xaux_interp));
        im = reshape(colors, n_1d_aux_x, n_1d_aux_y, 3);

        if ~savefig
            %clf; imshow(im);
            %hold on;
            %plot(x0(:,1), x0(:,2), 'k.');
            axis off
            set( gca, 'XTick', [] );
            set( gca, 'yTick', [] );
            set( gca, 'XTickLabel', '' );
            set( gca, 'yTickLabel', '' );
            set(gcf, 'color', 'white');
            set(gca, 'YGrid', 'off');
        end


        if savefig
            imwrite(im, sprintf([basedir 'frame_%d.png'], frame), 'png' );
            %savepng(gcf, sprintf([basedir 'latent_coord_map_layer_%d'], l));
        end
        drawnow;
        frame = frame + 1;
        fprintf('.');
    end
end


end

% Kernel function
function sigma = se_kernel(x, y)
    if nargin == 0
        sigma = 'Normal SE covariance.'; return;
    end

    sigma = 0.9*exp( -0.5.*sq_dist(x, y));
end
