% Code to characterize the distribution of the singular values as a
% function of number of products of independent Gaussian matrices.
%
% We're sampling the derivatives of very deep Gaussian processes.
%
% David Duvenaud
% August 2013

function deep_singular_value_dist

D = 5;    % Dimension of matrices
L = 50;   % Number of layers
n_samples = 100000;
savefigs = true;
scale = 1/3;
seed = 0;

addpath(genpath('utils'));

% Fix the seed of the random generators.
randn('state',seed);
rand('state',seed);

basedir = sprintf('../figures/spectrum/', seed);
mkdir(basedir);



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
        new_jacob = randn(D, D) .* scale;
        new_jacob_aug = [new_jacob, randn(D, D) .* scale];
        
        complete_jacob = new_jacob * complete_jacob;
        
        % Augment complete Jacobian with identity to denote that the input is
        % being fed in.
        complete_jacob_c = new_jacob_aug * [complete_jacob_c; eye(D)];
        %complete_jacob_c = complete_jacob_c + new_complete_jacob_c;
                
        lambdas(l, i, :) = svd( complete_jacob );  % Do SVD decomp.
        lambdas_c(l, i, :) = svd( complete_jacob_c );  % Do SVD decomp.
    end
end

% Show histograms of ratios of largest singular values to second-largest.
for l = 1:L
    figure(1); clf; 
    plot_hist(squeeze(lambdas(l, :, :))./repmat(squeeze(lambdas(l, :, 1))', 1, 5), D, l);
    if savefigs
        set_fig_units_cm( 8, 6 )
        save2pdf(sprintf([basedir, 'layer-%d'], l), gcf); 
    end
    figure(2); clf; %plot_hist(lambdas_c(l, :, :), D, l);
    plot_hist(squeeze(lambdas_c(l, :, :))./repmat(squeeze(lambdas_c(l, :, 1))', 1, 5), D, l);
    if savefigs
        set_fig_units_cm( 8, 6 )
        save2pdf(sprintf([basedir, 'con-layer-%d'], l), gcf); 
    end    
    %figure(3); clf; hist(min(lambdas(l, :, :), [], 3)./max(lambdas(l, :, :), [], 3), 100);
    %figure(4); clf; hist(min(lambdas_c(l, :, :), [], 3)./max(lambdas_c(l, :, :), [], 3), 100); title('c');
    %figure(5); clf; hist(min(lambdas(l, :, :), [], 3), 100);
    %figure(6); clf; hist(min(lambdas_c(l, :, :), [], 3), 100); title('c');
    
    %median(min(lambdas(l, :, :), [], 3 ))
    %median(min(lambdas_c(l, :, :), [], 3))
    %pause;
end

end




function plot_hist(cur_lambdas, D, l)
    n_bins = 50;

    hist_edges = linspace( 0, max(cur_lambdas(:)), n_bins);

    for d = 1:D
        subplot(1, D, d);
        % Make histogram
        [counts,bins] = hist(squeeze( cur_lambdas( :, d )), hist_edges ); %# get counts and bin locations
        barh(bins, counts, 'EdgeColor', 'none', 'BarWidth', 1)
        
        ylim([0, 1.1]);
        xlim([0, max(counts) .* 1.1]);

        set( gca, 'XTickLabel', '' );
        if d > 1
            set( gca, 'yTickLabel', '' );
        end

        sub_pos = get(gca,'position'); % get subplot axis position 
        set(gca,'position',sub_pos.*[1 1 1.1 1]) % stretch its width and height  
        
        set( gca, 'XTick', [] );
        xlabel(int2str(d))
    end
    %hist(ratios(l, :), 20);
    %imagesc(flipdim(cur_image, 1));
    %xlabel( 'Singular value #')
    %
    set(gcf, 'color', 'white');
    %title(['Depth ' int2str(l)], 'Interpreter', 'Latex', 'FontSize', 18);
    %set_fig_units_cm( 8, 6 )
end

% Plot hitogram using colors
function plot_hist_old(cur_lambdas, D, l)
    n_bins = 100;

    hist_edges = linspace( 0, max(cur_lambdas(:)), n_bins);
    

    cur_image = NaN( n_bins, D );
    for d = 1:D
        %hist_edges = linspace( 0, max(squeeze(cur_lambdas(:,:,d))), n_bins);
        
        % Make histogram
        counts = histc( squeeze( cur_lambdas( :, d )), hist_edges);
        cur_image( :, d) = counts ./ max(counts);
    end
    %hist(ratios(l, :), 20);
    imagesc(flipdim(cur_image, 1));
    xlabel( 'Singular value #')
    ylabel( 'Singular value distribution');
    set( gca, 'yTick', [] );
    %title(['Depth ' int2str(l)], 'Interpreter', 'Latex', 'FontSize', 18);
end

