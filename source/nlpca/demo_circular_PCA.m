% demo of 'circular PCA' (NLPCA.cir) 
% using the inverse network architecture with circular units
% 
% data: noisy circle
%
% circular units provide components as closed curves 
%
% see also: Scholz et al. BIRD conference, Berlin 2007
%           www.nlpca.org

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create circular data

    num=100;                      % number of samples
    t=linspace(-pi , +pi , num);  % angular value t=-pi,...,+pi
    data=zeros(2,numel(t));      % data set
    data(1,:)=sin(t);
    data(2,:)=cos(t);
    data = data + randn(size(data))*0.2; % add noise
    % plot(data(1,:),data(2,:),'.')


% start component extraction

  [c,net,network]=nlpca(data,1,  'type',          'inverse'  ,...
                                 'circular',      'yes'      ,...
                                 'max_iteration', 500        );

% save result

  % save nlpca_result_circle   net network

    
%  plot component

    nlpca_plot(net)
    title('{\bf Circular PCA}')
    axis equal
    