function demo_missing_data
% Nonlinear PCA for missing data estimation.
%
% artificial data set: helix loop
% No sample is complete! In each sample one of the three values is missing.
%
% see also: Scholz et al. Bioinformatics, 2005
%           www.nlpca.org

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create data: helix loop

    num=1000;                      % number of samples
    t=linspace(-pi , +pi , num);     % angular value t=-pi,...,+pi
    data_noisefree=[sin(t);cos(t); t]; % helix loop
    data_noised=data_noisefree+0.05*randn(size(data_noisefree));% add noise
    % plot3(data_noised(1,:),data_noised(2,:),data_noised(3,:),'.')
    % view(95,60)
    
    
% remove values in order to get an artificial missing data set

    % create index 'idx_nan' for missing data ('0'-m
    [tmp,idx]=sort(rand(size(data_noised)));
    idx_nan=(idx~=1);  
    % At both ends of the helix loop only values of x- or y-axis
    % are removed otherwise there would not be a unique solution.
    % (If z would be missing, the remaining x,y coordinates would 
    % almost be the same at both the start and the end of the loop.)
    idx_nan(:,1:5)=ones; 
      idx_nan(1,1)=0;idx_nan(2,2)=0;idx_nan(2,3)=0;
      idx_nan(1,4)=0;idx_nan(3,5)=0;
    idx_nan(:,end-4:end)=ones;       
      idx_nan(3,end-4)=0;idx_nan(2,end-3)=0;idx_nan(1,end-2)=0;
      idx_nan(2,end-1)=0;idx_nan(1,end)=0;

    data_nan=data_noised;
    data_nan(idx_nan==0)=NaN; % remove selected values from noised data set 

    
% component extraction

    [c,net,network]=nlpca(data_nan,1,'type','inverse',...
                                     'units_per_layer',[1,12,3],... % hid=12, to be more flexible
                                     'max_iteration',2000);
    
    
% plot component

    nlpca_plot(net); hold on
    plot3(data_noised(1,:),data_noised(2,:),data_noised(3,:),'.')

    
% missing data estimation error

    data_recon=nlpca_get_data(net);
    e = (data_recon-data_noisefree).^2;
    MSE = mean(e(isnan(data_nan))); 
    fprintf(1,'\n Mean square error (MSE) of missing data estimation: %0.4f\n',MSE);
    
    
% save result

    % save nlpca_result   net network data_recon

