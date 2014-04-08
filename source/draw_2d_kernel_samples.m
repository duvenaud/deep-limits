function draw_2d_kernel_samples
% Make figures of additive kernels in 2 dimensions
%
% David Duvenaud
% April 2014
% =================

addpath(genpath([pwd '/../']))
addpath('utils/');
clear all
close all

% generate a grid
range = -4:0.1:4;
[a,b] = meshgrid(range, range);
xstar = [a(:) b(:)];


% Define a sqexp kernel
% ==========================
covfunc = {'covSEiso'};
hyp.cov = log([1,1]);
K = feval(covfunc{:}, hyp.cov, xstar, xstar);
n = length( xstar);
K = K + diag(ones(n,1)).*0.000001;
cholK = chol(K);


% Plot draws from a that prior
% ================================================

for i = 1:10;
    seed=i;
    randn('state',seed);
    rand('state',seed);

    y = cholK'*randn(n, 1);
    figure;
    surf(a,b,reshape(y, length( range), length( range) ), 'EdgeColor','none','LineStyle','none','FaceLighting','phong'); 
    nice_figure_save(['../figures/two-d-draws/sqexp-draw-', int2str(i)]);
    
    fprintf('.');
end

end


function nice_figure_save(filename)

%    axis off
    set(gcf, 'color', 'white');
    set( gca, 'xTickLabel', '' );
    set( gca, 'yTickLabel', '' );    
    set( gca, 'zTickLabel', '' );    
    set(gca, 'TickDir', 'in')

    tightfig;
    set_fig_units_cm(12,12);
    
    myaa('publish');
    savepng(gcf, filename);
    %filename_eps = ['../figures/additive/3d-kernel/', filename, '.eps']
    %filename_pdf = ['../figures/additive/3d-kernel/', filename, '.pdf']
    %print -depsc2 filename_eps
    %eps2pdf( filename_eps, filename_pdf, true);
end
