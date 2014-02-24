function plot_deep_kernel

% This file is a demo of the kernel which extends
% the 2-layer rbf network kernel from Youngmin Cho's thesis
% http://cseweb.ucsd.edu/~yoc002/paper/thesis_youngmincho.pdf
% Eqn 4.4
%
% This version connects x to every layer.
% 
% Feb 2014
% David Duvenaud


connected = false;  % Does the input connect to every layer
seed=5;
savefigs = false;
N = 200;
x = linspace( -4, 4, N);   % Choose a set of x locations.
mu = zeros(N, 1);   % Set the mean of the GP to zero everywhere.
layers = 3;



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
for layer = 1:layers
    for j = 1:N
        for k = 1:N
            sigma(j,k, layer) = deep_covariance( x(j), x(k), layer - 1 );
        end
    end
    fprintf('.');
end


layers = layers + 1;
sigma(:,:, layers) = 1;

realdeep = true
if realdeep
    for j = 1:N
        for k = 1:N
            sigma(j,k, layers) = deep_covariance( x(j), x(k), 1000 );
        end
    end
end

for j = 1:N
    for k = 1:N
        sigma_ou(j,k) = ou_covariance( x(j), x(k));
    end
end


figure(1); clf;
%for layer = 1:layers
plot(x, squeeze(sigma_ou(100,:)), 'b-', 'Linewidth', 1); hold on;
    plot(x, squeeze(sigma(100,:,4)), 'r-', 'Linewidth', 1); hold on;
    
    ylim( [0, 1.1] );
    
    
%end
xlabel ('x - x''');
ylabel( 'cov( f(x), f(x'')');
legend( { 'OU kernel', 'deep kernel'}, 'Location', 'Best' );


set_fig_units_cm( 14, 10 );
%axis tight
%set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
fig_title = 'Deep RBF Network Kernel Functions';
title(fig_title, 'Interpreter', 'Latex', 'FontSize', 12);

if savefigs
    set(gcf, 'color', [.99 .99 .99]);
    save2pdf(sprintf('../figures/deep_kernel_connected'), gcf);  
    myaa('publish');
    savepng(gcf, sprintf('../figures/deep_kernel_connected'));  
end


figure(2); clf;
for layer = 1:layers
    f(:, layer) = mvnrnd( mu, sigma(:,:,layer) ); % Draw a sample from a multivariate Gaussian.
end
plot(x, f, 'Linewidth', 1);      hold on;           % Plot the drawn value at each location.
set_fig_units_cm( 14, 10 );
fig_title = 'Draws from a GP with Deep RBF Kernels';
title(fig_title, 'Interpreter', 'Latex', 'FontSize', 12);
legend( {'1 layer', '2 layers', '3 layers', '\infty layers'}, 'Location', 'Best' );

if savefigs
    set(gcf, 'color', [1 1 1]);
    save2pdf(sprintf('../figures/deep_kernel_connected_draws'), gcf);  
    myaa('publish');
    savepng(gcf, sprintf('../figures/deep_kernel_connected_draws'));  
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

function c = ou_covariance(x, y)
    c = exp( - abs( x - y ));
end
