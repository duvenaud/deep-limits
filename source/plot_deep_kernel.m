function plot_deep_kernel

% This file is a demo of the kernel which extends
% the 2-layer rbf network kernel from Youngmin Cho's thesis
% http://cseweb.ucsd.edu/~yoc002/paper/thesis_youngmincho.pdf
% Eqn 4.4
%
% This version connects x to every layer.
% 
% Sept 2013
% David Duvenaud


connected = false;  % Does the input connect to every layer
seed=5;
savefigs = true;
N = 200;
rangeleft = -3;
rangeright = 5;
x = linspace( rangeleft, rangeright, N);   % Choose a set of x locations.
mu = zeros(N, 1);   % Set the mean of the GP to zero everywhere.
layers = 3;
infinity = 100;

fig_units_x = 10;
fig_units_y = 5;

if connected
    basedir = sprintf('../figures/deep_kernel_seed%d/', seed);
else
    basedir = sprintf('../figures/deep_kernel_connected_seed%d/', seed);
end

addpath('utils');
% Fix the seed of the random generators.
randn('state',seed);
rand('state',seed);

% Specify the covariance between function values.
sigma = NaN(N,N,layers);
sigmaplot = NaN(N,layers);
for layer = 1:layers
    for j = 1:N
        for k = 1:N
            sigma(j,k, layer) = deep_covariance( x(j), x(k), layer - 1 );
        end
        sigmaplot(j, layer) = deep_covariance( x(j), 0, layer - 1 );
    end
    fprintf('.');
end


layers = layers + 1;
sigma(:,:, layers) = 1;

realdeep = true
if realdeep
    for j = 1:N
        for k = 1:N
            sigma(j,k, layers) = deep_covariance( x(j), x(k), infinity );
        end
        sigmaplot(j, layers) = deep_covariance( x(j), 0, infinity );
    end
end
clf;

figure(1); clf;
% First, plot some dashed lines to get the correct handles.

%clf;

for layer = 1:layers-1
    %plot(x, squeeze(sigmaplot(:,layer)), '--', 'Linewidth', 1, 'Color',colorbrew(layer));
    dashline(x, squeeze(sigmaplot(:,layer)), 1.2,1,1.2,1, 'Linewidth', 1, 'Color',colorbrew(layer));
    hold on;
end
for layer = 1:layers-1
    h(layer) = plot(0,0 , '--', 'Linewidth', 1, 'Color',colorbrew(layer));
    hold on;
end
h(layers) = plot(x, squeeze(sigmaplot(:,layers)), 'k-', 'Linewidth', 1); hold on;

xlabel ('x - x''');
ylabel( 'cov( f(x), f(x'')');
xlim([rangeleft, rangeright]);
ylim( [0, 1.1] );
set_fig_units_cm( fig_units_x, fig_units_y );
legend( h, {'1 layer', '2 layers', '3 layers', '\infty layers'}, 'Location', 'Best' );
legend boxoff
%axis tight
%set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
%fig_title = 'Deep RBF Network Kernel Functions';
%title(fig_title, 'Interpreter', 'Latex', 'FontSize', 12);

if savefigs
    set(gcf, 'color', [.99 .99 .99]);
    filename = sprintf('../figures/deep_kernel_connected');
    save2pdf(filename, gcf, 300, true);
    %myaa('publish');
    %savepng(gcf, sprintf('../figures/deep_kernel_connected'));  
end


figure(2); clf;
for layer = 1:layers - 1
    f(:, layer) = mvnrnd( mu, sigma(:,:,layer) ); % Draw a sample from a multivariate Gaussian.
    % Plot the drawn value at each location.
    dashline(x, f(:, layer), 1.2,1,1.2,1, 'Linewidth', 1, 'Color',colorbrew(layer));
    hold on;
end
f(:, layers) = mvnrnd( mu, sigma(:,:,layers) ); % Draw a sample from a multivariate Gaussian.
plot(x, f(:, layers), 'k-', 'Linewidth', 1); hold on;           % Plot the drawn value at each location.

xlim([rangeleft, rangeright]);
ylim([-2.9, 2.9]);
set_fig_units_cm( fig_units_x, fig_units_y );
%fig_title = 'Draws from a GP with Deep RBF Kernels';
%title(fig_title, 'Interpreter', 'Latex', 'FontSize', 12);
%legend( {'1 layer', '2 layers', '3 layers', '\infty layers'}, 'Location', 'Best' );
%legend boxoff

if savefigs
    set(gcf, 'color', [1 1 1]);
    save2pdf(sprintf('../figures/deep_kernel_connected_draws'), gcf, 300, true);  
    %myaa('publish');
    %savepng(gcf, sprintf('../figures/deep_kernel_connected_draws'));  
end



end

% Squared-exp covariance function:
function c = covariance(x, y)
    c = exp( - 0.5 * ( (( x - y )) .^ 2 ) );
end

function cxy = deep_covariance(x, y, layers)
    cxy = covariance(x, y);
    cxx = 1;%covariance(x, x);
    cyy = 1;%covariance(y, y);
    for l = 1:layers
        cxy = exp( cxy - 0.5*(cxx + cyy) -0.5*(x-y).^2);
    end
end
