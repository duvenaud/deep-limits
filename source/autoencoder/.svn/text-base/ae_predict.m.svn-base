% Classifies based on an autoencoder
%
% David Duvenaud
% Feb 2009
% ==========================================

function hat_labels = ae_predict( ae, digits, test_data, max_num_classes )

num_digits = length(digits );

[n, blah] = size( test_data );

cross_entropies = zeros( n, num_digits );

% copy dataset without labels
test_data_nolabels = test_data;
test_data_nolabels( :, end - max_num_classes + 1:end ) = zeros( n, max_num_classes );

for d = 1:num_digits
    cur_test_data = test_data_nolabels;
    
    % replace the label
    cur_test_data( :, end - max_num_classes + digits(d) + 1 ) = ones( n, 1 );
    
    hidden = sigmoid( cur_test_data * ae.weights + repmat(ae.hidden_bias', n, 1 ));       
    visible = sigmoid( hidden * ae.weights' + repmat(ae.visible_bias', n, 1 ));

    data = test_data_nolabels( :, end - max_num_classes + 1: end );
    visible = visible(  :, end - max_num_classes + 1:end );      
    cross_entropies( :, d ) = -sum( data .* log( visible ) + ( 1 - data ) .* log( 1 - visible ), 2);        
end


% choose the label with the min cross entropy
[ignore, best_ix] = max( cross_entropies, [], 2 );        
hat_labels = digits(best_ix);     
end
        
        
    