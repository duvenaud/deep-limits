% demo of nonlinear PCA (NLPCA) using the inverse network architecture
% data: noisy circle
%
% Note, that with inverse architecture the component is able to intersect 
%
% see also: Scholz et al. Bioinformatics, 2005
%           www.nlpca.org

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create circular data

    num=100;                      % number of samples
    t=linspace(-pi , +pi , num);  % angular value t=-pi,...,+pi
    data=[sin(t);cos(t)];         % circle
    data = data + 0.2*randn(size(data)); % add noise
    % plot(data(1,:),data(2,:),'.')


    
% component extraction

    [c,net,network]=nlpca(data,1,'type','inverse',  'max_iteration',3000);

    
    
% plot component

    nlpca_plot(net)
    title('{\bf NLPCA inverse}')
    axis equal
 
    
% save result

    % save nlpca_result_circle   net network
