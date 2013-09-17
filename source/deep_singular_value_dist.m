% Code to characterize the distribution of the singular values as a
% function of number of products of independent Gaussian matrices.
%
% David Duvenaud
% August 2013

D = 10;    % Dimension of matrices
L = 10;   % Number of layers
n_samples = 4000;

lambdas = NaN(L, n_samples, D);
for i = 1:n_samples
    J = eye(D); 
    for l = 1:L;
        % Generate a random matrix.
        A = randn(D);
        J = J*A;
        
        % Do SVD decomp
        lambdas(l, i, :) = svd( J );
    end
    %fprintf('.');
end

% Show histograms of ratios of largest singular values to second-largest.

n_bins = 100;

for l = 1:L
    
    cur_image = NaN( n_bins, D );
    
    cur_lambdas = lambdas(l, :, :);
    hist_edges = linspace( 0, max(cur_lambdas(:)), n_bins);
        
    for d = 1:D
        % Make histogram
        counts = histc( squeeze( lambdas(l, :, d )), hist_edges);
        cur_image( :, d) = counts ./ max(counts);
    end
    %hist(ratios(l, :), 20); 
    imagesc(flipdim(cur_image, 1));
    xlabel( 'singular value #')
    ylabel( 'singular value distribution');
    set( gca, 'yTick', [] );
    title(['Depth ' int2str(l)]);
    pause;
    %median( ratios(l, :) )
end