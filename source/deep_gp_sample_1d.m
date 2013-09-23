function deep_gp_sample_1d
%
% Plot the latent coordinates that points are mapped to by a deep function
%
% David Duvenaud
% Sept 2013

connected = true;  % Does the input connect to every layer
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

layers = 140;
n_1d = 400;
n_1d_aux = 1500;
savefig = false;

addpath(genpath('utils'));

x0 = linspace( -1, 1, n_1d)';
x0aux = linspace( -1, 1, n_1d_aux )';


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



for l = 1:layers
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
    
    sigma = se_kernel(aug', aug') + eye(size(aug,1)) * 1e-6;
    %figure(2); imagesc(K)

    k_x_xaux = se_kernel(aug', augaux');

    % Now sample second deepest layer.
    x = mvnrnd( mu, sigma)';

    % Work out warping distribution conditional on the already sampled points.
    mucond = k_x_xaux' / sigma * x;
    %sigmacond = k_x_xaux' / sigma * k_x_xaux;

    % Now sample some more points, conditioned on those.
    %xaux(:,1) = mvnrnd( mucond(:,1), sigmacond);
    %xaux(:,2) = mvnrnd( mucond(:,2), sigmacond);

    % R = RandOrthMat(2);  % Neural network version
    % xaux = 10./(1 + exp(-xaux*R));

    %xaux = xaux + mucond;
    xaux = mucond;  
    %figure(1000); 
    clf;
    
    
    %aux_ix_red = xaux(:,1) < 0; 
    %aux_ix_blue = xaux(:,1) >= 0;
    %plot(x0aux(aux_ix_red,1), x0aux(aux_ix_red,2), 'r.');
    %hold on;
    %plot(x0aux(aux_ix_blue,1), x0aux(aux_ix_blue,2), 'b.');

    plot( x0aux, xaux, 'b- '); hold on;
    plot( x0, x, 'kx' );
    
    fig_title = sprintf('Layer %d', l);
    title(fig_title, 'Interpreter', 'Latex', 'FontSize', 18);
    pause;
end


end


function sigma = se_kernel(x, y)
    if nargin == 0
        sigma = 'Normal SE covariance.'; return;
    end

    sigma = 0.5.*exp( -0.5.*sq_dist(x, y).*3);
end
