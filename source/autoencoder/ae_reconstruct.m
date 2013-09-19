% Sends the input through the network to see what it sees the data as
%
% David Duvenaud
% Feb 2009
% ==========================================

function visible = ae_reconstruct( ae, data )

[n, d ] = size( data );

hidden = sigmoid( data * ae.weights + repmat(ae.hidden_bias', n, 1 ));       
visible = sigmoid( hidden * ae.weights' + repmat(ae.visible_bias', n, 1 ));

for i = 1:n   
    subplot( 2, n, n + i );
    bits = data( i, 1:28*28 );
    bits = reshape( bits, 28, 28 );
    imagesc( bits );
    set(gca,'xtick',[],'ytick',[])
    
    subplot( 2, n, i );
    bits = visible( i, 1:28*28 );
    bits = reshape( bits, 28, 28 );
    imagesc( bits );
    set(gca,'xtick',[],'ytick',[])    
end


end