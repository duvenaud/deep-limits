% Code to characterize the distribution of the singular values as a
% function of number of products of independent Gaussian matrices.
%
% David Duvenaud
% August 2013

function deep_singular_value_dist

D = 10;    % Dimension of matrices
L = 30;   % Number of layers
n_samples = 4000;

lambdas = NaN(L, n_samples, D);
lambdas_c = NaN(L, n_samples, D);
for i = 1:n_samples
    
    % First layer only has D inputs.
    complete_jacob = randn(D); 
    complete_jacob_c = complete_jacob;
    lambdas(1, i, :) = svd( complete_jacob );  % Do SVD decomp.
    lambdas_c(1, i, :) = svd( complete_jacob_c );  % Do SVD decomp.
    for l = 2:L;
        % Generate a random 2DxD matrix for this layer's Jacobian
        new_jacob = randn(D, D);
        new_jacob_aug = randn(D, D);
        
        complete_jacob = new_jacob*complete_jacob;
        
        % Augment complete Jacobian with identity to denote that the input is
        % being fed in.
        new_complete_jacob_c = [complete_jacob_c, eye(D)] * [new_jacob; new_jacob_aug];
        complete_jacob_c = complete_jacob_c + new_complete_jacob_c;
                
        lambdas(l, i, :) = svd( complete_jacob );  % Do SVD decomp.
        lambdas_c(l, i, :) = svd( complete_jacob_c );  % Do SVD decomp.
    end
end

% Show histograms of ratios of largest singular values to second-largest.
for l = 1:L
    figure(1); clf; plot_hist(lambdas(l, :, :), D, l);
    figure(2); clf; plot_hist(lambdas_c(l, :, :), D, l);
    pause;
end

end

function plot_hist(cur_lambdas, D, l)
    n_bins = 100;

    hist_edges = linspace( 0, max(cur_lambdas(:)), n_bins);

    cur_image = NaN( n_bins, D );
    for d = 1:D
        % Make histogram
        counts = histc( squeeze( cur_lambdas(1, :, d )), hist_edges);
        cur_image( :, d) = counts ./ max(counts);
    end
    %hist(ratios(l, :), 20);
    imagesc(flipdim(cur_image, 1));
    xlabel( 'singular value #')
    ylabel( 'singular value distribution');
    set( gca, 'yTick', [] );
    title(['Depth ' int2str(l)]);
    
end
