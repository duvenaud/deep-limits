function ae_rep

addpath('utils');
addpath('nlpca');

% Fix the seed of the random generators.
seed = 0;
randn('state',seed);
rand('state',seed);

scale = 0.5;
N = 200;
%angles = linspace(9*pi/6,(11/6)*pi,N)';
angles = linspace(4*pi/6,(11/6)*pi,N)';

input_noise = 0.0005;
true_output_noise = 0.01;
model_output_noise = 0.01;

x = (angles.*cos(angles) + randn(N,1).*0.01).* scale;
y = (angles.*sin(angles) + randn(N,1).*0.01).* scale;

x = x + randn(size(x)) .* sqrt(input_noise);
y = y + randn(size(y)) .* sqrt(input_noise);

angles = (angles - mean(angles));

figure(1);
plot(x+y,y-x, '.');
%xlims = xlim;
%ylims = ylim;

xlims(1) = min(x) - 0.1*(max(x) - min(x));
xlims(2) = max(x) + 0.1*(max(x) - min(x));
ylims(1) = min(y) - 0.1*(max(y) - min(y));
ylims(2) = max(y) + 0.1*(max(y) - min(y));



data = [x,y];

%[c,net]=nlpca(data', 2, 'type', 'bottleneck', 'units_per_layer', [2, 15 , 2 , 15, 2]);
%figure(10); imagesc(c);

n_1d = 600;
xrange = linspace( xlims(1), xlims(2), n_1d);
yrange = linspace( ylims(1), ylims(2), n_1d);

% Make a grid
[xa, xb] = ndgrid( xrange, yrange ); %randn(n, D);   % Deepest latent layer.
xgrid = [ xa(:), xb(:) ];

%pc = nlpca_get_components(net,xgrid');

%pc = nlpca_get_components(net,data');

sigma = se_kernel( data', data') + eye(N).*model_output_noise;
k_x_xaux = se_kernel(data', xgrid');

% Work out warping distribution conditional on the already sampled points.
pc = k_x_xaux' / sigma * [angles, randn(size(angles)).*true_output_noise];
pc = pc + 0.4;  % offset to change the color scheme.

% rotate
%pc = [pc(:,1) + pc(:,2), pc(:,1) - pc(:,2)];

colors = coord_to_color(sin(2.*pi.*pc*2));
im = reshape(colors, n_1d, n_1d, 3);

% plot the data on the image with anti-aliasing.
for i = 1:N
    closest_x = round((x(i) - xlims(1))/(xlims(2) - xlims(1)) * n_1d);
    closest_y = round((y(i) - ylims(1))/(ylims(2) - ylims(1)) * n_1d);
    im(closest_x,closest_y,:) = 1;
    im(closest_x,closest_y + 1,:) = 1;
    im(closest_x,closest_y - 1,:) = 1;
    im(closest_x - 1,closest_y,:) = 1;
    im(closest_x + 1,closest_y,:) = 1;
    im(closest_x + 2,closest_y,:) = 0.5*(1 + im(closest_x + 2,closest_y,:));
    im(closest_x - 2,closest_y,:) = 0.5*(1 + im(closest_x - 2,closest_y,:));
    im(closest_x,closest_y + 2,:) = 0.5*(1 + im(closest_x,closest_y + 2,:));
    im(closest_x,closest_y - 2,:) = 0.5*(1 + im(closest_x,closest_y - 2,:));
    im(closest_x + 1,closest_y - 1,:) = 0.5*(1 + im(closest_x + 1,closest_y - 1,:));
    im(closest_x - 1,closest_y + 1,:) = 0.5*(1 + im(closest_x - 1,closest_y + 1,:));
    im(closest_x + 1,closest_y + 1,:) = 0.5*(1 + im(closest_x + 1,closest_y + 1,:));
    im(closest_x - 1,closest_y - 1,:) = 0.5*(1 + im(closest_x - 1,closest_y - 1,:));
end

figure(4);
imshow(im);
%set_fig_units_cm( 8,8 );
%if savefig
imwrite(im, 'hidden_bluegreen.png', 'png' );
%savepng(gcf, [basedir 'layer_0']);
%end
end

function sigma = se_kernel(x, y)
    if nargin == 0
        sigma = 'Normal SE covariance.'; return;
    end

    sigma = 0.5.*exp( -0.5.*sq_dist(x, y)*2);
end
