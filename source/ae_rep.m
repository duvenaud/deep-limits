clear all;
close all;

addpath('utils');
addpath('nlpca');

% Fix the seed of the random generators.
seed = 0;
randn('state',seed);
rand('state',seed);

scale = 0.5;
N = 50;
angles = linspace(pi/6,pi + pi/2,N)';
x = (angles.*cos(angles) + randn(N,1).*0.01).* scale;
y = (angles.*sin(angles) + randn(N,1).*0.01).* scale;

figure(1);
plot(x,y, '.');
xlims = xlim;
ylims = ylim;


data = [x,y];

[c,net]=nlpca(data', 2, 'type', 'bottleneck', 'units_per_layer', [2, 15 , 2 , 15, 2]);
figure(10); imagesc(c);

n_1d = 300;
xrange = linspace( xlims(1), xlims(2), n_1d);
yrange = linspace( ylims(1), ylims(2), n_1d);

% Make a grid
[xa, xb] = ndgrid( xrange, yrange ); %randn(n, D);   % Deepest latent layer.
xgrid = [ xa(:), xb(:) ];

pc = nlpca_get_components(net,xgrid');

%pc = nlpca_get_components(net,data');

colors = coord_to_color(sin(2.*pi.*pc'*10));
im = reshape(colors, n_1d, n_1d, 3);
figure(4);
imshow(im);
%set_fig_units_cm( 8,8 );
%if savefig
%imwrite(im, 'hidden.png', 'png' );
%savepng(gcf, [basedir 'layer_0']);
%end
