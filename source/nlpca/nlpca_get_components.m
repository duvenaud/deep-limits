function pc = nlpca_get_components(net,data)
% pc = nlpca_get_components(net,data) % extracting components 'pc' from new data
% pc = nlpca_get_components(net)      % extracting components 'pc' from train data
%
% using network architecture 'net'
% from [pc, net, network] = nlpca(data,k)
%
% pc   : row-wise are the (nonlinear) principal components z
%        columns are the samples
% data : rows are the original variables (attributes)
%        columns are the samples
%
% See also: nlpca_get_data nlpca nlpca_plot
% 

%
%  This file is part of the NLPCA package.
%
%  The NLPCA package is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2 of the License, or
%  (at your option) any later version.
% 
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
% 
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software
%  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
%
%  Copyright (C) 2006-2008 Matthias Scholz
%  http://www.nlpca.org/matlab.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global FIXED_WEIGHTS 
global SILENCE
global GROUP
if isempty(SILENCE),  SILENCE=false; end


if ~isfield(net,'weight_matrices') 
  net=reduce_parameters(net);
end

fct=net.functions_per_layer;
W=net.weight_matrices;

if nargin==1
  if ~isempty(net.data_class) % set GROUP labels
    GROUP=net.data_class;
  elseif net.version >= 0.71, % set GROUP=[] only for new versions
    GROUP=[];
  end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(net.type,'bottleneck')
   if nargin==1
     data=net.data_train_in;     
   end

   % check data
       num_units=net.units_per_layer(1);
       if size(data,1) ~= num_units
        s1=['dimension of data (',num2str(size(data,1)),') must be equal to '];
        error([s1,'number of units in first layer (',num2str(num_units),')'])
       end
       if isnan(sum(sum(data)))
        error('nlpca_get_components: bottleneck type does not work with NaN''s')
       end


   % propagation of data through part I of the network
  
       S=data;
       S=feval(fct(1,:),S);
       S_bias=ones(1,size(S,2));

       for i=1 : net.component_layer-1
         S=[S_bias;S];
         S=feval(fct(i+1,:),W{i}*S);
       end
       pc=S;
       
       if strcmp(net.circular,'yes'), pc=net_pq2phi(pc); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(net.type,'inverse')
  if nargin==1
    pc=net.data_train_in;
  end

  if nargin==2

     % check data
       num_units=net.units_per_layer(end);
       if size(data,1) ~= num_units
        s1=['dimension of data (',num2str(size(data,1)),') must be equal to '];
        error([s1,'number of units in last layer (',num2str(num_units),')'])
       end

  % if ~SILENCE
    fprintf(1,'# estimating optimal input\n'); 
  % end;
                                                                               
  w=[];  
  for i=1:length(net.units_per_layer)-1
    % w=[w;reshape(net.weight_matrices{i},prod(size(net.weight_matrices{i})),1)];
    w=[w;reshape(net.weight_matrices{i}, numel(net.weight_matrices{i}) ,1)];
  end
  FIXED_WEIGHTS=w;  % constant weights of the network

    tmp=struct('data_train_out'     ,data,...
               'mode'               ,'symmetric',...
               'type'               ,'inverse',...
               'units_per_layer'    ,net.units_per_layer,...
               'functions_per_layer',net.functions_per_layer,...
               'component_layer'    ,1,...
               'circular'           ,net.circular,...
               'circular_idx'       ,net.circular_idx,...
               'weight_decay'       ,'no',...
               'pre_scaling'        ,'no',...
               'plotting'           ,'no',...
               'silence'           ,'yes',...
               'max_iteration'      ,50);

    % tmp = train_network(tmp);
                                                                               
    best_result= zeros(net.units_per_layer(1), size(data,2));
    e_best     = repmat(inf,1,size(data,2));
    num_corrected  = size(data,2);
    n=0;
                                                                               
    while ((num_corrected > 0.05*size(data,2)) && (n<100 )) || (n<20)
      % use 20 runs, if then correction is still needed: use upto 100 runs
      tmp = train_network(tmp);
      components = tmp.data_train_in;
      out = nlpca_get_data(net,components);
       %e   = mean((out-data).^2, 1);  % does not work with NaN's 
       e=(out-data).^2;
       e(isnan(e))=0;
       e=sum(e);
      idx=(e_best > e);
      best_result(:,idx) = components(:,idx);
      e_best(idx) = e(idx);
      num_corrected=sum(idx);
      n=n+1;
      if ~SILENCE
        fprintf(1,'# run %i: %i replaced component values\n',n,num_corrected);
      end
      tmp.data_train_in=[];
    end
                                                                               
    pc=best_result;
  end
end

FIXED_WEIGHTS=[];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=circ(a)
% old: function [a,circ_r]=circ(a)
%
% circular units
%
% [z,r]=circ(a)
%

%  Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global CIRCULAR_INDEX
global CIRCULAR_R  % for back propagation

CIRCULAR_R=zeros(size(CIRCULAR_INDEX,2),size(a,2));

for c=1:size(CIRCULAR_INDEX,2)    
  % p=a(CIRCULAR_INDEX(1,c),:);
  % q=a(CIRCULAR_INDEX(2,c),:);
  % r=sqrt(p^2 + q^2)
  r=sqrt(sum(a(CIRCULAR_INDEX(:,c),:).^2));
  % p=p/r; q=q/r;
  a(CIRCULAR_INDEX(1,c),:)=a(CIRCULAR_INDEX(1,c),:) ./ r;
  a(CIRCULAR_INDEX(2,c),:)=a(CIRCULAR_INDEX(2,c),:) ./ r;
  CIRCULAR_R(c,:)=r;
end

% circ_r=CIRCULAR_R; % old

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fs_phi]=net_pq2phi(fs_pq)
%
% fs_phi=net_pq2phi(fs_pq)
% 

% function [fs_phi,circ_idx]=net_pq2phi(fs_pq)

global CIRCULAR_INDEX

% [fs_pq,r]=circ(fs_pq);
fs_pq=circ(fs_pq); % make sure to lie on the unit circle (r==1)

fs_phi=fs_pq;


for c=1:size(CIRCULAR_INDEX,2)    
  p=fs_pq(CIRCULAR_INDEX(1,c),:);
  q=fs_pq(CIRCULAR_INDEX(2,c),:);
  t=atan2(q,p); % with t in the interval (-pi ... pi)
  fs_phi(CIRCULAR_INDEX(1,c),:)=t; % p <- t
end
    

fs_phi(CIRCULAR_INDEX(2,:),:)=[];


% circ_idx
 %   for c=1:size(CIRCULAR_INDEX,2)  
 %     c_idx=zeros(size(fs_pq,1),1);
 %     c_idx(CIRCULAR_INDEX(:,c),:)=1;
 %   end
 %   c_idx(CIRCULAR_INDEX(2,:))=[];
 %   circ_idx=find(c_idx==1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% explanation:
%
% t=-pi:0.01:pi;
% x=cos(t);
% y=sin(t);
% plot(x,y);
% axis([-1.2,1.2,-1.2,1.2])
%
% t=atan2(y,x) with t in (-pi   ...pi  )  is similar to  
% t=atan(y./x) with t in (-pi/2 ...pi/2)
% but takes into account the quadrant in which (x,y) lies
% 
% atan2(y,x) contrasts with atan(y/x), whose results are 
% limited to the interval (-pi   ...pi  )
%
% x = r * cos(t)	
% y = r * sin(t)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function fs_pq=net_phi2pq(fs_phi)
%
% fs_pq=net_phi2pq(fs_phi)
%
% 

global CIRCULAR_INDEX


% enlarge units in component layer
    for c=1:size(CIRCULAR_INDEX,2)   
      fs_phi=fs_phi([1:CIRCULAR_INDEX(1,c),CIRCULAR_INDEX(1,c):end],:);
    end


  fs_pq=fs_phi;

  for c=1:size(CIRCULAR_INDEX,2)    
    t=fs_phi(CIRCULAR_INDEX(1,c),:);
    p=cos(t);
    q=sin(t);
    fs_pq(CIRCULAR_INDEX(1,c),:)=p; 
    fs_pq(CIRCULAR_INDEX(2,c),:)=q;
  end   



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% explanation:
%
% t=-pi:0.01:pi;
% x=cos(t);
% y=sin(t);
% plot(x,y);
% axis([-1.2,1.2,-1.2,1.2])
%
% t=atan2(y,x) with t in (-pi   ...pi  )  is similar to  
% t=atan(y./x) with t in (-pi/2 ...pi/2)
% but takes into account the quadrant in which (x,y) lies
% 
% atan2(y,x) contrasts with atan(y/x), whose results are 
% limited to the interval (-pi   ...pi  )
%
% x = r * cos(t)	
% y = r * sin(t)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x=linr(x)
% returns the input 
% to be used as transfer-function in the same way as 'tanh' or 'circ'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function net=reduce_parameters(h)
% net=reduce_parameters(network)
%
% reduces the complete algorithm parameter settings in 'network' to a 
% simple basic set ('net') used to apply the network, not to optimise it. 
% All pre-processing steps will be included to the weight matrices
%
% This 'net'-set can be applied (as well as the larger 'network' set) 
% to new data to get the corresponding component values,
% and also to new component values for reconstructing the original data. 
% 
%   pc   = nlpca_get_components(net,data)
%   data = nlpca_get_data(net,pc)
%  

% Author   : Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global NET

  net=struct('data_train_in',      [],... 
             'data_train_out',     [],...
             'data_class',         [],...  % [1,1,1,1, 2,2,2,2, 3,3,3,3]
             'mode',               [],...  % { symmetric | hierarchic }
	         'type',               [],...  % { inverse | bottleneck }
             'units_per_layer',    [],...  % [ 3  ,  5   ,  2   ,  5 ,    3   ]
             'functions_per_layer',[],...  % [     'tanh';'linr';'tanh';'linr']
             'component_layer',    [],...  % 3 (layer id)
             'circular',           [],...  % yes, no
             'circular_idx',       [],...  % [1,0,0]
             'weight_matrices',    [],...  % W{1} ... W{4}
             'version',            []);    % 0.71

  net.data_train_in =h.data_train_in;
  net.data_train_out=h.data_train_out;
  if h.version >= 0.71, 
    net.data_class=h.data_class; 
    net.version=h.version;
  end;

  net.mode=h.mode;
  net.type=h.type;
  net.units_per_layer=h.units_per_layer;
  net.functions_per_layer=h.functions_per_layer;
  net.component_layer=h.component_layer;
  net.circular=h.circular;
  net.circular_idx=h.circular_idx;

  NET=h.units_per_layer;
  if strcmp(h.circular,'yes') 
    NET(h.component_layer)=NET(h.component_layer)+sum(h.circular_idx);
  end

  W=vector2matrices(h.weight_vector);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pre_scaling
  if strcmp(h.pre_scaling,'yes')
     if ~isempty(h.scaling_factor) 
          % ...*W = c*X  -> ...1/c *W = X
          W{end}=1/h.scaling_factor * W{end};
        if strcmp(h.type,'bottleneck')
          % W* (c*X) + bias -> (W*c) * X + bias
          W{1}(:,2:end)=h.scaling_factor * W{1}(:,2:end);       
        end
     end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre_pca

  if strcmp(h.pre_pca,'yes')

        w    = W{end}(:,2:end);
        bias = W{end}(:,1);

            % w*hid + bias = ev (X-m)
            % pinv(ev)*w*hid + pinv(ev)*bias + m = X
            %   w_new    = pinv(ev)*w
            %   bias_new = pinv(ev)*bias + m
            % w_new + bias_new = X
        bias_new = h.inverse_eigenvectors * bias + h.pca_removed_mean;
        w_new = h.inverse_eigenvectors * w;

        %W{end}(:,2:end) = w_new;
        %W{end}(:,1)     = bias_new;
        W{end} = [bias_new,w_new];   % korrected march 2006
        net.units_per_layer(end)=size(w_new,1);

    if strcmp(h.type,'bottleneck')
        w    = W{1}(:,2:end);
        bias = W{1}(:,1);

            % hid = w * ev * (X-m) + bias
            % hid = -w*ev*m + bias + w*ev*X
            %   bias_new = bias - w * ev * m
            %   w_new    = w * ev
            % hid = w_new * x + bias_new
        bias_new = bias - w * h.eigenvectors * h.pca_removed_mean; 
        w_new    = w * h.eigenvectors;

        %W{1}(:,2:end) = w_new;
        %W{1}(:,1)     = bias_new;
        W{1} = [bias_new,w_new];   % korrected march 2006
        net.units_per_layer(1)=size(w_new,2); % korrected 1->2 (jan 2007)
    end
  end  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre_unit_variance

  if strcmp(h.pre_unit_variance,'yes')

        w    = W{end}(:,2:end);
        bias = W{end}(:,1);

            % w*hid + bias               = (x-m)./s
            % ss=repmat(s,1,size(w,2))
            % ss.*w * hid + s.*bias + m  =  x
            %   w_new    = ss.*w 
            %   bias_new = s.*bias + m
            % w_new * hid + bias_new     =  x 
        bias_new = h.removed_std.*bias + h.removed_mean;
        %w_new   = meshgrid(h.removed_std,randn(1,size(w,2)))'.*w;
        w_new    = repmat(h.removed_std,1,size(w,2)).*w;

        W{end}(:,2:end) = w_new;
        W{end}(:,1)     = bias_new;

    if strcmp(h.type,'bottleneck')
        w    = W{1}(:,2:end);
        bias = W{1}(:,1);

            % w*((x-m)./s) + bias
            % ss=repmat(s',size(w,1),1)
            % w./ss * x - w*(m./s) + bias
            %   w_new    = w./ss *
            %   bias_new = bias - w*(m./s)
            % w_new * x + bias_new
        bias_new = bias - w*(h.removed_mean./h.removed_std);
        % w_new  = w ./ meshgrid(h.removed_std,randn(1,size(w,1)));
        w_new    = w ./ repmat(h.removed_std',size(w,1),1);

        W{1}(:,2:end) = w_new;
        W{1}(:,1)     = bias_new;
    end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% circular: angular shift (so far only for inverse type)

if strcmp(net.circular,'yes') && strcmp(net.type,'inverse') 

  % shift of component values
      fs=net.data_train_in(net.circular_idx(1),:);
      phi_shift=-pi-fs(1);
      newfs=fs+phi_shift;
      newfs(newfs<-pi)=newfs(newfs<-pi)+2*pi;
      sn=sign(newfs*[1:1:size(newfs,2)]'); % direction of circle
      net.data_train_in(net.circular_idx(1),:)=sn*newfs;


  % matrix rotation
     % create circular_index of first circular unit
     pos=find(net.circular_idx==1,1);
     circular_index=[pos;pos+1];

     WW=W{net.component_layer};
     w=WW(:,circular_index+1); % +1 because of bias
     rot=[cos(-phi_shift),-sin(-phi_shift);sin(-phi_shift),cos(-phi_shift)];
     w=w*rot;
     w(:,2)=sn*w(:,2);% switch sign of y-axis (start/end still keeps at(-1,-1))
     WW(:,circular_index+1)=w;
     W{net.component_layer}=WW;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  net.weight_matrices=W;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [h,net]=train_network(h_in)
%
% Artificial Neural Network (MLP) for performing non-linear PCA.
% Network is trained by using the data and parameters of struct h.xxx 
%
% h=train_network(h)
% [h,net]=train_network(h)  'net' contains the pure network parameters
%                            pre-processing is included
%                           'h' contains all parameters, 
%                            including training parameters
%                           'h' is the internal variable name for 'network'
%
% h=struct(parameter_1 , value_1,...
%          parameter_n , value_n);
%
%<parameter>        <value>
%data_train_out    output train data 
%                  samples are column-wise, variables are in different rows
%                  in case of NaN, h.type will be set to 'inverse'
%data_train_in     input train data (default:h.data_train_out) 
%data_weight_out   weight matrix, weights each single error
%                  if ('NaN' in data): element in weight matrix will set to '0'
%                  Example: 'data_weight_out',[0.7,0.8, 0;0.2, 1, 0.9],...  
%	               default: [] or '0' if NaN; [0,1,1;1,1,1] if [NaN,x,x;x,x,x]
%                  data_weight_out can be a [num*n] vector for sample weights 
%data_test_out     output test data
%data_test_in      input  test data
%data_class        sample group identifiers for plotting in different colours
%                  default: [];   Example: [1,1,1,1, 2,2,2,2, 3,3,3,3] 
%    
%mode              'symmetric' : symmetric NLPCA  (classical autoencoder)
%                  'hierarchic': hierarchic NLPCA (default if components>1)
%type              'bottleneck': standard bottleneck autoencoder, 
%                     default if no NaN in input data (data_train_in)
%                     NaN in data_train_out are possible
%	               'inverse'   : Inverse training, only output data are given. 
%                     Network weights and input data are estimated.
%                     default, if only one data set and NaN's 
%                     in data_train_out
%                     components are able to intersect in inverse mode
%circular          circular units(nodes) in component layer, default: 'no'
%circular_idx      example: [1,0,0] the first of three units is circular
%                  (a circular unit regards as one single node even
%                   though it is performed internally by two units) 
%weighted_data     weighting errors, by using matrix h.data_weight_out
%                  default:'yes' if ~isempty(h.data_weight_out)
%                  default:'yes' if NaN in h.data_train_out
%                  default:'no'  else
%number_of_components  default (k): if units_per_layer is defined then
%                    k=h.units_per_layer(h.component_layer) else
%                    k=1 if dim==2, k=2 if dim==3, k=3 if dim>3,
%plotting          plotting while training 
%                  default: 'yes'
%units_per_layer   network units, example: [3, 5 , 2 , 5 , 3]
%                  3 input and output units, 2 component units
%                  2x5 hidden units for nonlinearity
%                  default: [dim_in (4+4xf) f (4+2xf) dim_out]
%functions_per_layer Example: ['tanh';'linr';'tanh';'linr'],layer 2,3,...,end
%component_layer     id of component layer, default(type=inverse):1
%                  default(type=bottleneck): middle layer 
%hierarchic_layer  id of hierarchic layer, default: id of component_layer
%hierarchic_idx    default: [[1;0],[1;1],[0;1]], see 'get_hierarchic_idx.m'
%hierarchic_coefficients  Coefficients of each term in the error function
%                  [c(1),c(1:2),c(1:3),...,c(1:n-1),c(2:n)]
%                  default: [1,1,1,...,1,0.01]
%sort_components     default:'yes' if hierarchic mode
%                  default:'no'  if symmetric mode
%weight_vector     all network weights in one vector 
%weight_initialization  default: 'set_weights_randomly'
%                  alternative: 'set_weights_linear'
%opti_algorithm    weight optimization, default:'cgd'
%error_function    default:'error_hierarchic.m' if hierarchic mode
%                  default:'error_symmetric.m'  if symmetric mode
%gradient_function'default:'derror_hierarchic.m' if hierarchic mode
%                  default:'derror_symmetric.m'  if symmetric mode
%weight_decay      'yes': weight_decay is on (default)
%                  'no' : weight_decay is off
%weight_decay_coefficient   value between 0 and 1, default: 0.001 
%max_iter          number of itarations
%                  default: min([5*numel(h.data_train_out),3000])
%classification    data_train_out have to be class labels
%                  default:'no' 
%connected_network_idx  In case of several connected autoencoder networks
%pass              In case of additional training runs, pass: 1,2,3,...
%
%pre_scaling       yes: limit the max std in the data set to keep the network 
%                       in the linear range at begin
%                  default: set std to 0.1, if no 'scaling_factor' is specified
%scaling_factor    0.xx: Scaling multiplier
%                  default: 0.1/max(std(data),[],2)
%pre_unit_variance {'yes','no'} default:'no', unit variance normalization
%removed_mean      mean for reconstruction
%removed_std       std  for reconstruction
%pre_pca           {'yes','no'} default:'no', PCA preprocessing, the first n 
%                  components are used, n is the number of output units
%pca_components    complete components, in case of NaN's in 'train_out'
%pca_residual_train_error  train error of (n+1:end) components  
%pca_residual_test_error   test error of (n+1:end) components 
%pca_removed_mean          data mean
%eigenvectors              nxnum matrix for extraction
%inverse_eigenvectors      numxn matrix for recontruction
%     
%stepwise_error    save error in each iteration in
%                  h.train_error and h.test_error, default:'yes'
%printing_error    default:'no' 
%train_fuction_error  error in each iteration of h.error_function
%train_error       train error in each iteration
%test_error        test  error in each iteration
%best_test_weights weights of minimal test error
%video             animation sequence, {'yes','no'}, default:'no'
%video_weights     weight-vectors for each iteration step
%video_iter        selected iterations for video sequence
%
%silence           switch off any comments

% Author   : Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global GROUP  % used in nlpca_plot.m
global NET
% global FKT % old
global FCT
global ITMAX
global WEIGHT_DECAY
% global SPECIAL % old for totally connected network
global CHAIN
global DATADIST
global WEIGHTED_DATA
global COMPONENT_LAYER
global E_TRAIN_FUNCTION E_TRAIN E_TEST 
global TRAIN_IN TRAIN_OUT    % normelized data
global TEST_IN  TEST_OUT     % normelized data
global BEST_TEST_WEIGHTS
global SORT_COMPONENTS
global F_COUNT; F_COUNT=0;
global CLASSIFICATION
global HIERARCHIC_MODE
global HIERARCHIC_VAR
global HIERARCHIC_LAYER
global HIERARCHIC_IDX
global PLOTTING 
global VIDEO VIDEO_WEIGHTS  VIDEO_ITER 
global SAVE_ERROR PRINT_ERROR
global INVERSE
global FIXED_WEIGHTS
global CIRCULAR
global CIRCULAR_IDX
global CIRCULAR_INDEX
global CIRCULAR_R
global SILENCE


NLPCAversion=0.88;


  h=struct('data_train_in',[],...            %  1
           'data_train_out',[],...           %  2
           'data_weight_out',[],...           
           'data_test_in',[],...             %  3
           'data_test_out',[],...            %  4
           'data_class',[],...               % [1,1,1,2,2,2,3,3,3]
           ...
           'mode',[],...                     % 24 symmetric, hierarchic
           'type',[],...                     % 75 inverse, bottleneck
           'circular',[],...                 % yes, no
           'circular_idx',[],...             % [1,0,0]
           'weighted_data',[],...            % 21 yes, no 
           'number_of_components',[],...       % {5,26} [], 2
           'plotting',[],...                 % 60 yes, no
           'units_per_layer',[],...          %  5 [3, 5 , 2 , 5 , 3];
           'functions_per_layer',[],...      %  6 ['linr';'tanh';'linr';'tanh';'linr']
           'component_layer',[],...            % 26 [], 3
           'hierarchic_layer',[],...         % 25 [], 3
           'hierarchic_idx',[],...           % 32 [[1;0],[1;1],[0;1]]
           'hierarchic_coefficients',[],...  % 30 [], [c1,c2,0.01]
           'sort_components',[],...            % 29 yes, no
           'weight_vector',[],...            % 15 [], w
           'weight_initialization',[],...    % 55 set_weights_linear
           'opti_algorithm',[],...           %  8 cgd_alg
           'error_function',[],...           %  9 error_fuction
           'gradient_function',[],...        % 10 derror_function
           'weight_decay',[],...             % 13 yes, no 
           'weight_decay_coefficient',[],... % 13 0.01
           'max_iteration',[],...            % 11 [], 300
           'classification',[],...           % 31 yes, no 
           'connected_network_idx',[],...    % 22 [5,9,13]
           'pass',[],...                     % 79 1,2,3,...
           ... 
           'pre_scaling',[],...              % 50 yes, no
           'scaling_factor',[],...           % 50 0.1 
           'pre_unit_variance',[],...        % 40 yes, no 
           'removed_mean',[],...             % 41 result
           'removed_std',[],...              % 42 result
           'pre_pca',[],...                  % 44 yes, no 
           'pca_components',[],...           %    result
           'pca_residual_train_error',[],... % 45 result
           'pca_residual_test_error',[],...  % 46 result
           'pca_removed_mean',[],...         % 47 result
           'eigenvectors',[],...             % 48 result
           'inverse_eigenvectors',[],...     % 49 result
           ...
           'stepwise_error',[],...           % 61 yes, no 
           'printing_error',[],...           % 61 yes, no 
           'train_function_error',[],...     % 16 result
           'train_error',[],...              % 17 result
           'test_error',[],...               % 18 result
           'best_test_weights',[],...        % 27 result
           'video',[],...                    %    yes, no 
           'video_weights',[],...            %    result
           'video_iter',[],...               %    result
           ...
           'silence',[],...                  % yes, no 
           'version',[],...                  % 0.6   
           'date',[]);                       % 11-Dec-2006






% fill full struct h with fields of h_in 
  s=fieldnames(h_in);
  for i=1:length(s), 
    if isfield(h,s{i}), 
      h=setfield(h,s{i},getfield(h_in,s{i})); 
    else 
      error(['Reference to non-existent field: .',s{i}])
    end
  end
  
  % h=check_syntax(h); % check is moved to nlpca.m

% silence
    if ~strcmp(h.silence,'yes');
      h.silence='no'; 
    end
    SILENCE = strcmp(h.silence,'yes');
% date and version
    h.date=date; 
    h.version=NLPCAversion;
    if ~SILENCE, fprintf(1,'\nNonlinear PCA (NLPCA) \n# version: %1.2f\n',h.version), end
% labels
    if ~isempty(h.data_class)
      GROUP=h.data_class; 
    else
      GROUP=[];
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set defaults for:
% NaN, h.type, h.data_weight_out, h.weighted_data, DATADIST, 

    % move data_train_in to data_train_out if data_train_out is empty
    % only data_train_out is used when both data_in and data_out are identical
    if isempty(h.data_train_out) && ~isempty(h.data_train_in)
        h.data_train_out=h.data_train_in;
        h.data_train_in=[];
        if ~SILENCE, fprintf(1,'# move ''.data_train_in'' to ''.data_train_out''\n'), end
      % move data_in to data_out in case of NaN or of inverse type
      % if sum(sum(isnan(h.data_train_in)))>0 
      %   h.data_train_out=h.data_train_in;
      %   h.data_train_in=[];
      %   if ~SILENCE, fprintf(1,'# move ''.data_train_in'' to ''.data_train_out'' because of missing data\n'), end
      % end
      % if strcmp(h.type,'inverse')
      %   h.data_train_out=h.data_train_in;
      %   h.data_train_in=[];
      %   if ~SILENCE, fprintf(1,'# move ''.data_train_in'' to ''.data_train_out'' because of .type=''inverse''\n'), end
      % end
    end

    num_NaN=0;
    if ~isempty(h.data_train_out)
      idx_NaN=isnan(h.data_train_out);
      num_NaN=sum(sum(idx_NaN));
    end

    if isempty(h.type), h.type='bottleneck'; end    

    % data (sample) weight vector -> data weight matrix
    if size(h.data_weight_out,1)==1 && size(h.data_train_out,1)>1
       h.data_weight_out=meshgrid(h.data_weight_out,...
                                  zeros(size(h.data_train_out(:,1))) );
    end

    if num_NaN > 0 && ~strcmp(h.pre_pca,'yes');
      if ~SILENCE, 
       fprintf(1,'# detecting NaN in ".data_train_out", using "data_weight"\n')
      end
      if isempty(h.data_weight_out)
        h.data_weight_out=ones(size(h.data_train_out));
      end
      h.data_weight_out(idx_NaN)=zeros;
    end
    
    if isempty(h.weighted_data),
      if ~isempty(h.data_weight_out)
        h.weighted_data='yes'; else h.weighted_data='no';
      end
    end

    if (num_NaN > 0) && isempty(h.data_train_in) && ~strcmp(h.type,'inverse')
     if ~SILENCE, fprintf(1,'# set ".type" to "inverse", because of NaN\n');end
     h.type='inverse';
    end

    % Data
    if strcmp(h.type,'bottleneck')
      if isempty(h.data_train_in), h.data_train_in=h.data_train_out; end
    end

    if ~SILENCE, fprintf(1,'# network  type: %s\n',h.type); end

 
TRAIN_IN  = h.data_train_in;
TRAIN_OUT = h.data_train_out; 
TEST_IN   = h.data_test_in;
TEST_OUT  = h.data_test_out; 
WEIGHTED_DATA=strcmp(h.weighted_data,'yes');
if strcmp(h.type,'inverse'),    INVERSE=1; end
if strcmp(h.type,'bottleneck'), INVERSE=0; end
DATADIST  = h.data_weight_out;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set defaults for network architecture:
% h.mode
% h.number_of_components, h.units_per_layer, h.component_layer, h.hierarchic_layer


  if isempty(h.number_of_components)  % set to dim-1, but max to 3 
     if isempty(h.units_per_layer)
        ld=size(h.data_train_out,1);
        h.number_of_components=min(ld-1,3); 
     end
  end

  if isempty(h.units_per_layer)
     ld=size(h.data_train_out,1);
     lf=h.number_of_components;
     if lf<10, lh=(2+2*lf); else lh=lf; end             
     if strcmp(h.type,'inverse'),    h.units_per_layer=      [lf,lh,ld]; end
     if strcmp(h.type,'bottleneck'), h.units_per_layer=[ld,lh,lf,lh,ld];end
  end

  if isempty(h.component_layer)
    if strcmp(h.type,'inverse'),    h.component_layer=1; end
    if strcmp(h.type,'bottleneck')
      h.component_layer=ceil(length(h.units_per_layer)/2);
    end
  end

  if ~strcmp(h.circular,'yes'), 
    h.circular='no'; 
  else
    h.circular='yes';
  end
  if isempty(h.circular_idx) && strcmp(h.circular,'yes') 
    % if h.number_of_components==1, h.circular_idx=1;     end
    % default: h.circular_idx=[1,0,0]
    h.circular_idx=zeros(1,h.number_of_components);
    h.circular_idx(1)=1;
  end

  if isempty(h.functions_per_layer) % new FCT
     % s=1*['tanh'] ->  116    97   110   104
     func_num=length(h.units_per_layer);
     s = char(meshgrid([116,97,110,104],ones(1,func_num)));
     %if h.component_layer>1, s(h.component_layer-1,:)='linr'; end
     s(1,:)='linr';
     s(h.component_layer,:)='linr';
     s(end,:)='linr';
     if strcmp(h.circular,'yes'); s(h.component_layer,:)='circ'; end
     h.functions_per_layer=s;
  end

  if isempty(h.number_of_components)
     h.number_of_components=h.units_per_layer(h.component_layer);
  end



  if isempty(h.mode), 
    if h.number_of_components==1
      h.mode='symmetric';
    else
      h.mode='hierarchic'; 
    end
  end

  if strcmp(h.mode,'hierarchic'),
    if ~SILENCE, fprintf(1,'# training mode: hierarchic\n'); end
    if isempty(h.hierarchic_layer), h.hierarchic_layer=h.component_layer; end
    if isempty(h.error_function), h.error_function='error_hierarchic'; end
    if isempty(h.gradient_function),h.gradient_function='derror_hierarchic';end
    num_hierarchic_components=h.units_per_layer(h.hierarchic_layer);
    if isempty(h.hierarchic_coefficients), 
       h.hierarchic_coefficients=[ones(1,num_hierarchic_components), 0.01]; end
    if isempty(h.sort_components), h.sort_components='yes'; end
    if ( strcmp(h.sort_components,'yes') && h.number_of_components==1 ),
       h.sort_components='no'; end
    if isempty(h.hierarchic_idx), 
       h.hierarchic_idx=get_hierarchic_idx(num_hierarchic_components); end
    if ~SILENCE, fprintf(1,'# hierarchic layer: %i \n',h.hierarchic_layer); end
    if ~SILENCE, fprintf(1,'# hierarchic coefficients: %s\n',...
            num2str(h.hierarchic_coefficients,' %0.5g')); end
  else 
    if ~SILENCE, fprintf(1,'# training mode: symmetric\n'); end
    if isempty(h.error_function), h.error_function='error_symmetric'; end
    if isempty(h.gradient_function),h.gradient_function='derror_symmetric';end
    h.hierarchic_coefficients=0;
    h.hierarchic_layer=0;
    if isempty(h.sort_components), h.sort_components='no'; end
    if ( strcmp(h.sort_components,'yes') && h.number_of_components==1 ),
       h.sort_components='no'; end
  end
  if strcmp(h.circular,'yes')
       h.sort_components='no'; % does not work yet with circular units
  end

HIERARCHIC_MODE  = strcmp(h.mode,'hierarchic');
HIERARCHIC_LAYER = h.hierarchic_layer;
HIERARCHIC_VAR   = h.hierarchic_coefficients;
HIERARCHIC_IDX   = h.hierarchic_idx;
SORT_COMPONENTS    = strcmp(h.sort_components,'yes');
NET              = h.units_per_layer;  
  if ~SILENCE, 
    fprintf(1,'# number of components: %i\n',h.number_of_components);
    fprintf(1,'# network architecture: [');
    fprintf(1,'%i-',h.units_per_layer(1:end-1));
    fprintf(1,'%i]\n',h.units_per_layer(end));  
  end
% old: FKT              = h.functions_per_layer(2:end,:); % old
FCT              = h.functions_per_layer;
if ~isempty(h.connected_network_idx) 
  CHAIN          = [1, h.connected_network_idx ,length(h.units_per_layer)];
end
COMPONENT_LAYER  = h.component_layer;
CIRCULAR         = strcmp(h.circular,'yes');
if CIRCULAR, 
  % set CIRCULAR_INDEX
    cidx=h.circular_idx;
    pos=find(cidx==1);
    for i=1:size(pos,2)
      cidx=cidx([1:pos(i)+(i-1),pos(i)+(i-1):end]);
    end
    cnidx=find(cidx==1);
    CIRCULAR_INDEX = reshape(cnidx,2,size(pos,2));
    CIRCULAR_IDX   = h.circular_idx;
  NET(h.component_layer)=NET(h.component_layer)+sum(h.circular_idx);
  CIRCULAR_R = [];
  if HIERARCHIC_MODE % enlarge HIERARCHIC_IDX
    for i=1:size(CIRCULAR_INDEX,2)   
      HIERARCHIC_IDX=HIERARCHIC_IDX([1:CIRCULAR_INDEX(1,i),CIRCULAR_INDEX(1,i):end],:);
    end
  end
  if ~SILENCE, fprintf(1,'# using circular units\n'); end
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set defaults for training parameters:
% h.weight_decay, h.max_iteration, h.classification, h.opti_algorithm
 
if isempty(h.weight_decay), h.weight_decay='yes'; end
if strcmp(h.weight_decay,'yes')
  if isempty(h.weight_decay_coefficient), h.weight_decay_coefficient=0.001; end
end

if isempty(h.max_iteration)
  h.max_iteration=min([5*numel(h.data_train_out),3000]);
end

if isempty(h.classification), h.classification='no'; end

if isempty(h.opti_algorithm), h.opti_algorithm='opti_alg_cgd'; end


if strcmp(h.weight_decay,'yes')
  WEIGHT_DECAY=h.weight_decay_coefficient;
   if ~SILENCE, fprintf(1,'# weight decay: ON,  coefficient = %1.3g\n',h.weight_decay_coefficient); end
else
  WEIGHT_DECAY=0;
   if ~SILENCE, fprintf(1,'# weight decay: OFF\n'); end
end
ITMAX=h.max_iteration;
CLASSIFICATION=strcmp(h.classification,'yes');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% preprocessing, in each pass since h.data_xxx contains only the raw data
% h.pre_unit_variance, h.pre_pca, h.pre_scaling

if isempty(h.pass),  h.pass=1; else h.pass=h.pass+1; end

% h.pre_unit_variance
  if isempty(h.pre_unit_variance), h.pre_unit_variance='no'; end
  if strcmp(h.pre_unit_variance,'yes')
     if ~SILENCE, fprintf(1,'# normalising data to zero mean and std one in each variable (row-wise)\n'); end
     [h.removed_mean,h.removed_std,TRAIN_OUT,TRAIN_IN,TEST_OUT,TEST_IN]...
               = pre_unit_variance(TRAIN_OUT,TRAIN_IN,TEST_OUT,TEST_IN);
  end

% h.pre_pca
  if isempty(h.pre_pca), h.pre_pca='no'; end
  if strcmp(h.pre_pca,'yes')
      if num_NaN > 0 || size(TRAIN_OUT,1) > 1000
      %fprintf(1,'# switch PCA preprocessing off, due to NaN''s in data set\n')
      %h.pre_pca='no'; 
      if ~SILENCE, fprintf(1,'# PCA pre-processing by ''pca_network''\n'); end
      [h.pca_components,h.eigenvectors,h.inverse_eigenvectors,eval,...
       h.pca_removed_mean,tmp_xr,tmp_net] ... 
       = pca_network(TRAIN_OUT,h.units_per_layer(end),500);
       TRAIN_OUT=h.pca_components;
       if ~isempty(TRAIN_IN), TRAIN_IN=h.eigenvectors * ...
	(TRAIN_IN-repmat(h.pca_removed_mean,1,size(TRAIN_IN,2))); end;
       if ~isempty(TEST_OUT), TEST_OUT=h.eigenvectors * ...
        (TEST_OUT-repmat(h.pca_removed_mean,1,size(TEST_OUT,2))); end;
       if ~isempty(TEST_IN), TEST_IN =h.eigenvectors * ...
        (TEST_IN-repmat(h.pca_removed_mean,1,size(TEST_IN,2))); end;
     else
       if ~SILENCE, fprintf(1,'# PCA pre-processing by ''pre_pca''\n'); end
       [h.pca_removed_mean,h.eigenvectors,h.inverse_eigenvectors,...
       TRAIN_OUT,h.pca_residual_train_error,TRAIN_IN,tmp,...
       TEST_OUT,h.pca_residual_test_error,TEST_IN,tmp]...
       = pre_pca(h.units_per_layer(end),TRAIN_OUT,TRAIN_IN,TEST_OUT,TEST_IN);
     end
       if ~SILENCE, fprintf(1,'#  using first %i of %i PCA components\n',...
             h.units_per_layer(end),size(h.data_train_out,1)); end
  end

% h.pre_scaling, h.scaling_factor
if isempty(h.pre_scaling), h.pre_scaling='yes'; end
if strcmp(h.pre_scaling,'yes')
   if isempty(h.scaling_factor)
      tmp_std=zeros(1,size(TRAIN_OUT,2));
      for i=1:size(TRAIN_OUT,2)
        tmp=TRAIN_OUT(:,i);
        tmp_std(i)=std(tmp(~isnan(tmp)));
      end
      h.scaling_factor=0.1/max(tmp_std);
   end
end
if strcmp(h.pre_scaling,'yes')
    if ~SILENCE, fprintf(1,...
        '# data scaling, set max std to 0.1 (scal=%1.3g)\n',h.scaling_factor)
    end
   TRAIN_IN =h.scaling_factor*TRAIN_IN;
   TRAIN_OUT=h.scaling_factor*TRAIN_OUT;
   TEST_IN  =h.scaling_factor*TEST_IN;
   TEST_OUT =h.scaling_factor*TEST_OUT;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% weights
% h.weight_vector, h.weight_initialization 

% initializing weights, h.weight_vector
  if isempty(h.weight_vector) 
    if ~isempty(h.weight_initialization) 
     if ~SILENCE,
      fprintf(1,'# initializing weights using: %s.m\n',h.weight_initialization)
     end
      h.weight_vector=feval(h.weight_initialization,TRAIN_OUT);
    else
     if ~SILENCE,
      fprintf(1,'# initializing weights randomly: set_weights_randomly.m\n')
     end
      h.weight_initialization='set_weights_randomly';
      h.weight_vector=set_weights_randomly; 
    end
  end

% inverse training
  num_pattern=size(h.data_train_out,2);
  if strcmp(h.type,'inverse')
    if FIXED_WEIGHTS % estimating new components to given new data
      TRAIN_IN  = [];
      h.weight_vector = randn(NET(1)*num_pattern,1); % (',1' O.K.) * 0.1 ???
    else
      if h.pass==1 && strcmp(h.pre_pca,'no')
         train_in = randn(NET(1),num_pattern) * 0.1; end % O.K.
      if h.pass==1 && strcmp(h.pre_pca,'yes') % pca preproc. in the first step
         train_in = TRAIN_OUT(1:NET(1),:); end 
      if h.pass > 1
         train_in = h.data_train_in; end

      % set input data in front of weight vector 
      h.weight_vector=[reshape(train_in,NET(1)*num_pattern,1);...
                       h.weight_vector];
    end
  end

BEST_TEST_WEIGHTS=h.best_test_weights;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% error, plot and video 

% stepwise error calculating: h.stepwise_error, h.printing_error
    if isempty(h.stepwise_error), h.stepwise_error='yes'; end
    if isempty(h.printing_error), h.printing_error='no'; end

% h.plotting
    if isempty(h.plotting), h.plotting='yes'; end
 
% h.video
    if isempty(h.video), h.video='no'; end

E_TRAIN=[];
E_TEST =[];
E_TRAIN_FUNCTION =[];
SAVE_ERROR  = strcmp(h.stepwise_error,'yes');
PRINT_ERROR = strcmp(h.printing_error,'yes'); 
PLOTTING    = strcmp(h.plotting,'yes');
VIDEO       = strcmp(h.video,'yes');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%|||||||< optimization algorithm >|||||||||||||||||||||||||||||||||||||||||||||

h=check_semantic(h);

% h=get_best_initialization(h,300,50)
% recursive  h=get_best_initialization(h,50)

if ~SILENCE, fprintf(1,'# network training\n# ...\n\n'); end

h.weight_vector=feval(h.opti_algorithm,...
                h.weight_vector,h.error_function,h.gradient_function); 
            
if ~SILENCE, fprintf(1,'\n# network training - finished \n\n'); end
 
%||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save results

% h.train_error, h.test_error
  % error correction due to scaling
    if strcmp(h.pre_scaling,'yes')
      % if ~SILENCE, fprintf(1,'# correcting error due to data scaling\n'); end
      E_TRAIN=E_TRAIN * 1/h.scaling_factor.^2;
      E_TEST=E_TEST  * 1/h.scaling_factor.^2;
    end
  % error correction, add pca residual error
    if h.units_per_layer(end) < size(h.data_train_in,1)
      % if ~SILENCE, fprintf(1,'# correcting error, add the error of residual pca components\n'); end
      E_TRAIN = E_TRAIN * h.units_per_layer(end)/size(h.data_train_in,1)...
                + h.pca_residual_train_error;
      E_TEST  = E_TEST * h.units_per_layer(end)/size(h.data_train_in,1)...
                + h.pca_residual_test_error;
    end
  h.train_error        = [h.train_error ,E_TRAIN];
  h.test_error         = [h.test_error  ,E_TEST];
  h.train_function_error= [h.train_function_error,E_TRAIN_FUNCTION];

% inverse training - extracting estimated input data from weight vector
  if strcmp(h.type,'inverse')
    num_elements = NET(1)*size(h.data_train_out,2);
    h.data_train_in  = reshape(h.weight_vector(1:num_elements),...
                               NET(1), size(h.data_train_out,2));
    if CIRCULAR, h.data_train_in=net_pq2phi(h.data_train_in); end
    h.weight_vector  = h.weight_vector(num_elements+1 : end);
    VIDEO_WEIGHTS    = VIDEO_WEIGHTS(num_elements+1:end,:);
  end

h.video_weights = [h.video_weights,VIDEO_WEIGHTS];
h.video_iter    = [h.video_iter,VIDEO_ITER];
h.best_test_weights=BEST_TEST_WEIGHTS;


if nargout == 2
  net=reduce_parameters(h);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reconstruction
% (out=h.data_train_in;)
% h.inverse_eigenvectors*out/h.scaling_factor+ repmat(h.pca_removed_mean,1,num)
%
% transform new input 
% [tmp,num] = size(new_data);
% new_input = h.eigenvectors * (new_data - repmat(h.pca_removed_mean,1,num))...
%             * h.scaling_factor;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function idx=get_hierarchic_idx(hierarchic_num)
% idx=get_hierarchic_idx(hierarchic_num)
%
% Example (4 components): idx =
%
%     1     1     1     1     0
%     0     1     1     1     1
%     0     0     1     1     1
%     0     0     0     1     1
%

%%%%%%

idx=zeros(hierarchic_num,hierarchic_num+1);
for i=1:hierarchic_num
  idx(1:i,i)=1;
end

if hierarchic_num>1
  idx(2:end,end)=1; % for zero mean in component one
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function w=opti_alg_cgd(w, func, dfunc)
% Conjugate Gradient Algorithm
% (Polak-Ribiere expression)
%
%   w=opti_alg_cgd(w, func, dfunc)
%
% w     - initial weight vector
% func  - error function
% dfunc - gradient function
%

%  Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global TRAIN_IN TRAIN_OUT      % normelized data, only used in error function
global PLOTTING       
global SAVE_ERROR PRINT_ERROR
% global ITER;
global VIDEO VIDEO_WEIGHTS VIDEO_ITER 
global E_TRAIN E_TEST E_TRAIN_FUNCTION

global BEST_TEST_WEIGHTS; best_test=inf;
global SORT_COMPONENTS
global ITMAX 
global SILENCE

% for debugging only
  global DEBUG_PLOT  % plotting line search
         % DEBUG_PLOT=true; % true or false, for testing line search
         DEBUG_PLOT=false; 
        
        
        
        

  % first search direction dv (downhill: negative gradient)
    [dw,e]=feval(dfunc,w,TRAIN_IN,TRAIN_OUT);
    dv=-dw;

    
  if PLOTTING
    nlpca_plot(w,func,TRAIN_IN,TRAIN_OUT);  
    e_plot_last = e;
    drawnow;
  end
  VIDEO_WEIGHTS=[]; 
  VIDEO_ITER=[];
  if SORT_COMPONENTS, e_sort_last = e; end


  
  e_hist   = zeros(1,ITMAX);
  t_last=ones(1,6)*0.0001; % initialisation: step width of the last n steps
for i=1:ITMAX 

  % ITER=i; % global ITER, might be used somewhere else, or for debugging
 
  e_hist(i)=e;
  e_last = e; 

  
  %  if i==100, test_gradient(w, func, dfunc); end % for debugging


  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % line search in direction dv (downhill)

    % Depending on the application and state of optimisation, the step
    % width t can be on very different scales (e.g., 0.01 or 0.0001).
    % Thus, an initial step width is predicted by the last steps by using
    % the smallest t of the last steps, but not smaller than 1e-05.
    t_guess=max( min(t_last), 0.00001);
    [w, e, t]=line_search(w, dv, e, t_guess, func,TRAIN_IN,TRAIN_OUT); 
    t_last=[t_last(2:end),t]; % shift and add new t 
    % w   - minimum at direction dv of error e=E(w)
    
    [dw_new,e]=feval(dfunc,w,TRAIN_IN,TRAIN_OUT); % gradient of errorfuction
    % dw_new - gradient (positive gradient, uphill direction)
    % e      - error value E(w)
    
   
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % define new search direction dv (conjugate direction)
  
     % b1=dw_new'*dw_new; % Fletcher-Reeves
     b1=dw_new'*(dw_new-dw);  % Polak-Ribiere
     b2=dw'*dw;
     beta=b1/b2;
    
     dv=-dw_new+beta*dv; % new search direction dv
     dw=dw_new;
     
     
   % reset of conjugate search direction (set to negative gradient)
   if e > e_last, % may occure due to accuracy problems 
       dv=-dw_new; 
       if ~SILENCE, fprintf(1,'  # critical point of optimisation at iteration %i:\n    reset conjugate search direction to negative gradient\n',i); end
   end 
   % if (mod(i,50)==0), dv=-dw_new; end % optional reset every 50 iterations (error function might not be a ideal quadratic function)
     
   if isnan(e)  % NaN -> stop
     error('<opti_alg_cgd.m> square error is NaN, accuracy in line-search might be to small')
   end

   
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if SORT_COMPONENTS
   if (e / e_sort_last) < 0.90 || i==ITMAX || i==ITMAX-1 ||i==ITMAX-2
    e_sort_last = e; 
    if ~SILENCE, fprintf(1,'  # sorting components at iteration %i',i); end
    w=sort_components(w,TRAIN_IN,TRAIN_OUT);
    if ~SILENCE, fprintf(1,' - done\n'); end
   end
  end

  if PLOTTING || VIDEO
   if ( e / e_plot_last ) < 0.99 || i==ITMAX % or:  < 0.95  % or: if 1
    if PLOTTING
      nlpca_plot(w,func,TRAIN_IN,TRAIN_OUT); drawnow;        
    end
    if VIDEO
      VIDEO_WEIGHTS = [VIDEO_WEIGHTS,w]; 
      VIDEO_ITER = [VIDEO_ITER,i]; 
    end;
    e_plot_last = e;
   end
  end
 
  if (SAVE_ERROR || PRINT_ERROR)
   [E_train,E_test]=get_error(w,func); 
   E_TRAIN=[E_TRAIN,E_train'];
   E_TEST =[E_TEST,E_test'];
   if ~isempty(BEST_TEST_WEIGHTS)
    if E_test(end)<best_test
     BEST_TEST_WEIGHTS=w; 
     best_test=E_test(1); % PC1 only, if hierarchic 
    end
   end
  end;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
end % end of CGD iterations

E_TRAIN_FUNCTION = [e_hist,e];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [w_best, e_best, t_best]=line_search(w, dw, e0, t_guess, func, data_in, data_out)
%
% w  - weight vector
% dw - gradient: dE/dw
% e  - error: E(w)
%

%  Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The number of iterations is kept low, since is might be better to search
% in a new direction than searching too accurately in one direction.
% Accuracy is given by fixed iterations in golden section search. This 
% means that accuracy depends on step width: 
% (small step width -> small initial interval -> high accuracy).

global DEBUG_PLOT

iter_golden_section_search=6; % (works even with 3)
alpha=0.618034; % golden section coefficient
t=zeros(1,4);   % search positions on the line (fist, two middle, and last)
e=zeros(1,4);   % corresponding errors E(w(t))

if DEBUG_PLOT
    SUM_ITER=0;
    figure(2); hold off;
    tt=linspace(-0.001,0.001,100);   
    ee=zeros(size(tt));
    for i=1:numel(tt)
      ee(i)=feval(func,w+tt(i)*dw,data_in, data_out);
    end
    plot(tt,ee,'-'); hold on;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% expand search interval

t(1)=0; 
e(1)=e0;  % must be: e == feval(func,w+t(1)*dw,data_in, data_out) 
t(4)=t_guess;
e(4)=feval(func,w+t(4)*dw,data_in, data_out);       if DEBUG_PLOT, plot(t(1),e(1),'k.'); plot(t(4),e(4),'bo'); end

if e(4) > e(1) % got final interval, calculate t(2) and t(3)
   t(2)=t(1)+(1-alpha)*(t(4)-t(1));
   e(2)=feval(func,w+t(2)*dw,data_in, data_out);    if DEBUG_PLOT, plot(t(2),e(2),'bo'); end
   t(3)=t(1)+    alpha*(t(4)-t(1));
   e(3)=feval(func,w+t(3)*dw,data_in, data_out);    if DEBUG_PLOT, plot(t(3),e(3),'bo'); end
else % expand: add new t(4)
   t(3)=t(4);
   e(3)=e(4);
   t(4)=(1+alpha) * t(4);
   e(4)=feval(func,w+t(4)*dw,data_in, data_out);    if DEBUG_PLOT, plot(t(4),e(4),'bo'); end
   if e(4) > e(3) % got final interval, calculate t(2) 
      t(2)=t(1)+(1-alpha)*(t(4)-t(1));
      e(2)=feval(func,w+t(2)*dw,data_in, data_out); if DEBUG_PLOT, plot(t(2),e(2),'bo'); end
   else % expand: add new t(4)
     i=1;
     while e(4) < e(3) && i< 50 
       t(2)=t(3);
       e(2)=e(3); 
       t(3)=t(4);
       e(3)=e(4);
       t(4)=(1+alpha) * t(4);
       e(4)=feval(func,w+t(4)*dw,data_in, data_out); if DEBUG_PLOT, plot(t(4),e(4),'bo'); end
       i=i+1;
       if i==50; fprintf(1,'# line_search: to many iterations (>50) while expanding\n'); end;
     end
   end
end
   
if DEBUG_PLOT, plot(t,e,'r.'); end

if DEBUG_PLOT, [e;t], end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% golden section search

  if DEBUG_PLOT, plot(t(2),e(2),'kx');plot(t(3),e(3),'kx'); end

    for i=1:iter_golden_section_search
        if e(3) > e(2)
          t(4)=t(3);  % remove right value t(4)
          e(4)=e(3);
          t(3)=t(2);
          e(3)=e(2);
          t(2)=t(1)+(1-alpha)*(t(4)-t(1)); % split left interval
          e(2)=feval(func,w+t(2)*dw,data_in, data_out);  if DEBUG_PLOT, plot(t(2),e(2),'kx'); end
        else % if e(3) < e(2)
          t(1)=t(2);  % remove left t value t(1)
          e(1)=e(2);
          t(2)=t(3);
          e(2)=e(3);
          t(3)=t(1)+   alpha *(t(4)-t(1)); % split right interval
          e(3)=feval(func,w+t(3)*dw,data_in, data_out);  if DEBUG_PLOT, hold on; plot(t(3),e(3),'kx'); end
        end
    end     
    if e(2) < e(3), e_best=e(2); t_best=t(2); else e_best=e(3); t_best=t(3); end
 
  
    
  w_best=w+t_best*dw;
  
  
    if DEBUG_PLOT, plot(t_best,e_best,'r+'); hold off; figure(1); end     
    % if DEBUG_PLOT,  num_of_iterations=SUM_ITER-100, pause; end
    if DEBUG_PLOT, pause; end



function [E,n_out]=error_func(w,Pin,Pout)
% basic error function  
% [E,out]=error_func(w,Pin,Pout)
% 
% out - output of all nodes (inclusive hidden nodes!)
%
% WEIGHT_DECAY is set OFF
% WEIGHTED_DATA and DATADIST is set OFF
% NaN's (error by using missing data) are set to zero: E(isnan(E))=0;
%

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NET    % Example:  NET=[   2  ,   4  ,   1  ,   4  ,   2  ]
global FCT    % Example:  FCT=['linr';'tanh';'linr';'tanh';'linr']
% global WEIGHT_DECAY 
% global WEIGHTED_DATA DATADIST


W=vector2matrices(w);
[net_dim,net_num] = size(NET);
[Pin_dim,Pin_num] = size(Pin);

n_out = zeros(sum(NET),Pin_num); % n_out get the Outputs of all neurons
n_out(1:Pin_dim,:) = Pin;        % the input neurons at the first rows
n_out(1:Pin_dim,:) = feval( FCT(1,:) , Pin ); % in case of 'circ'


% forward propagation

  S_bias = ones(1,Pin_num);
 
  for i = 1:net_num-1
    if i==1, n_begin = 1;  else n_begin=sum(NET(1:i-1))+1;  end
    S_in = [S_bias;n_out(n_begin:sum(NET(1:i)),:)];
    n_out(sum(NET(1:i))+1:sum(NET(1:i+1)),:) = feval(FCT(i+1,:),W{i}*S_in);  
  end  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% error function

  Epattern = (n_out(sum(NET(1:end-1))+1:end,:) - Pout) .^2;

  Epattern(isnan(Epattern))=0;  % NaN: missing data

  % if WEIGHTED_DATA, Epattern=Epattern.*DATADIST; end

  E=0.5*sum(sum(Epattern));

  % E = E + WEIGHT_DECAY*0.5*sum(w.^2);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E,Sin]=error_symmetric(w,train_in,train_out)
% [E,out]=error_symmetric(w,train_in,train_out)

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NET    % Example:  NET=[   2  ,   4  ,   1  ,   4  ,   2  ]
global FCT    % Example:  FCT=['linr';'tanh';'linr';'tanh';'linr']
global WEIGHT_DECAY
global DATADIST
global WEIGHTED_DATA

global INVERSE         % second half only of the auto-associative network
global FIXED_WEIGHTS   % keep fixed all weight for searching optimal input

global CIRCULAR


if INVERSE
   if FIXED_WEIGHTS
     train_in = reshape( w , NET(1) , size(train_out,2) );
     w_train_in=w;  
     w=FIXED_WEIGHTS;
   else
     num_elements = NET(1)*size(train_out,2);
     train_in = reshape( w(1:num_elements) , NET(1) , size(train_out,2) );
     w_train_in=w(1:num_elements); % new <<<<<<<<<<<<<<<<<
     w   = w(num_elements+1 : end);
   end
end


W=vector2matrices(w);
[net_dim,net_num] = size(NET);
[train_in_dim,train_in_num] = size(train_in);


% forward propagation

  S_bias = ones(1,train_in_num);
  %Sin = train_in;  
  Sin = feval( FCT(1,:) , train_in ); % for 'circ' in INVERSE 


%if CIRCULAR
%  for i = 1:net_num-1
%    if i==COMPONENT_LAYER
%      for n=1:size(CIRCULAR_INDEX,2)
%        % r=sqrt(p^2 * q^2)
%        r=sqrt(sum(Sin(CIRCULAR_INDEX(:,n),:).^2));
%        % p=p/r; q=q/r;
%        Sin(CIRCULAR_INDEX(1,n),:)=Sin(CIRCULAR_INDEX(1,n),:) ./ r;
%        Sin(CIRCULAR_INDEX(2,n),:)=Sin(CIRCULAR_INDEX(2,n),:) ./ r;
%      end
%    end
%    Sin = [S_bias;Sin];
%    Sin = feval(FKT(i,:),W{i}*Sin);
%  end
%else % standard


  for i = 1:net_num-1
    Sin = [S_bias;Sin];
    Sin = feval(FCT(i+1,:),W{i}*Sin);
  end


%end 
   
%%%%%%%%%%%%%%%%%%%%%%%
% error function

  Epattern = (Sin - train_out).^2;

  Epattern(isnan(Epattern))=0;  % NaN: missing data

  if WEIGHTED_DATA, Epattern=Epattern.*DATADIST; end

  E=0.5*sum(sum(Epattern));

  E = E + WEIGHT_DECAY*0.5*sum(w.^2); % weight decay for real weights

  % smooth (0.01) weight decay also for input values
  % but not in case of circular units
  if INVERSE && ~CIRCULAR, E = E + 0.01*WEIGHT_DECAY*0.5*sum(w_train_in.^2); end % new <<<<<<<<<<<< 
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% line search test:
% 

%  global SUM_ITER

%     SUM_ITER=SUM_ITER+1;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dw,E,n_error,n_out]=derror_symmetric(w,train_in,train_out)
% gradient of error-fuction of symmetric NLPCA
%
% [dw,E,n_error,n_out]=derror_symmetric(w,train_in,train_out)
%
% dw - gradient (positive gradient, uphill direction)
% E  - error E(w)
%

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NET    % Example:  NET=[   2  ,   4  ,   1  ,   4  ,   2  ]
global FCT    % Example:  FCT=['linr';'tanh';'linr';'tanh';'linr']
global WEIGHT_DECAY
global DATADIST
global WEIGHTED_DATA

% CONNECT=logical(1); % historical: (CONNECT=2) full connected network, 

global INVERSE         % second half only of the auto-associative network
global FIXED_WEIGHTS   % keep fixed all weight for searching optimal input

global CIRCULAR
global CIRCULAR_INDEX  % used in circular PCA
global CIRCULAR_R


if INVERSE
   if FIXED_WEIGHTS
     train_in = reshape( w , NET(1) , size(train_out,2) );
     w_train_in=w;  
     w=FIXED_WEIGHTS;
   else
     num_elements = NET(1)*size(train_out,2);
     train_in = reshape( w(1:num_elements) , NET(1) , size(train_out,2) );
     w_train_in=w(1:num_elements); % new <<<<<<<<<<<<<<<<<
     w   = w(num_elements+1 : end);
   end
end


W=vector2matrices(w);
[net_dim,net_num] = size(NET);
[train_in_dim,train_in_num] = size(train_in);



% mean_of_input=mean(abs(w_train_in))
% mean_of_weights=mean(abs(w))
% mean_of_W1=mean(mean(abs(W{1})))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_out = zeros(sum(NET),train_in_num);  % n_out get the outputs of all units
%n_out(1:train_in_dim,:) = train_in;   % the input units at the first rows
n_out(1:train_in_dim,:) = feval( FCT(1,:) , train_in ); % for 'circ' 


%%%%%%%%%%%%%%%%%%%%%%%
% forward propagation

  S_bias = ones(1,train_in_num);
        
  for i = 1:net_num-1
    
    %if CONNECT == 1  % full-connected
      if i==1, n_begin = 1;  else n_begin=sum(NET(1:i-1))+1;  end
      Sa=n_out(n_begin:sum(NET(1:i)),:);
      S_in = [S_bias;Sa];
    %end
    
    % if CONNECT == 2  % total-connected  
    %   S_in = [S_bias;n_out(1:sum(NET(1:i)),:)];
    % end
    
    n_out(sum(NET(1:i))+1:sum(NET(1:i+1)),:) = feval(FCT(i+1,:),W{i}*S_in);  
    
  end  
  
%%%%%%%%%%%%%%%%%%%%%%%
% error function


  Epattern = (n_out(sum(NET(1:end-1))+1:end,:) - train_out ) .^2;

  Epattern(isnan(Epattern))=0;  % set error of missing data to zero

  if WEIGHTED_DATA, Epattern=Epattern.*DATADIST; end

  E=0.5*sum(sum(Epattern));

  E = E + WEIGHT_DECAY*0.5*sum(w.^2); % weight decay for real weights

  % smooth (0.01) weight decay also for input values
  % but not in case of circular units
  if INVERSE && ~CIRCULAR, E = E + 0.01*WEIGHT_DECAY*0.5*sum(w_train_in.^2); end % new <<<<<<<<<<<< 
  
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%< BACK PROPAGATION >%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  n_error = zeros(sum(NET),train_in_num);%n_error get the Errors of all neurons
  dW   = cell(1,net_num-1);           % Gradient
  W_bp = cell(1,net_num-1);           % special weight-matrix for backprob
                                      %   Important in case 'total connected'
                                      
  % Weight-matrices preparing for back propagation
  
  % if CONNECT == 1
      for u=1:net_num-1
        W_bp{u}=W{u}(:,2:NET(u)+1);  % cats the weights which belong to bias
      end  
  %  end
    
    %if CONNECT == 2
    % for u=1:net_num-1 % Weights have to be in an other structur backwards 
    %  for v=u:net_num-1  
    %   W_bp{u}=[W_bp{u} ; W{v}(:, sum(NET(1:u))-NET(u)+2 : sum(NET(1:u))+1 )];
    %  end
    % end    
    % end  
  
  
  % last layer
  
    S_tmp = n_out(end-NET(end)+1:end,:);
    %if     FCT(end,:) == 'tanh'    E_tmp = (1-S_tmp.^2).*(train_out-S_tmp); % old
    %elseif FCT(end,:) == 'linr'    E_tmp = train_out-S_tmp;   end           % old
    if     FCT(end,:) == 'tanh',    E_tmp = (1-S_tmp.^2).*(S_tmp-train_out);  % new <<<<<<<<<<<<<<<<<
    elseif FCT(end,:) == 'linr',    E_tmp = S_tmp-train_out;   end            % new <<<<<<<<<<<<<<<<<

    E_tmp(isnan(E_tmp))=0;                       % NaN: missing data
    if WEIGHTED_DATA, E_tmp=E_tmp.*DATADIST; end % weighted data 

    n_error(end-NET(end)+1:end,:) = E_tmp;
    
    
  % all other layers
   
    for n=1:net_num-1
      i=net_num-n;  % counts backwards 
      
      %E_tmp = n_error(sum(NET(1:i))+1 : sum(NET(1:i+1)),:); % already exist
      S_tmp = n_out(sum(NET(1:i-1))+1:sum(NET(1:i)),:);  
          
      % if CONNECT == 1
        dW{i} = E_tmp * [S_bias;S_tmp]';   % Gradient                   
      % end 
    
      % if CONNECT == 2
      %   dW{i} = E_tmp * [S_bias; n_out(1:sum(NET(1:i)),:) ]';   % Gradient 
      %   E_tmp = n_error(sum(NET(1:i))+1:end,:);  % with all layer behind
      % end
        

      if     FCT(i,:) == 'tanh'    E_tmp = (1-S_tmp.^2).*(W_bp{i}'*E_tmp);
      elseif FCT(i,:) == 'linr'    E_tmp = W_bp{i}'*E_tmp;   
      elseif FCT(i,:) == 'circ'    E_tmp = W_bp{i}'*E_tmp;   
        for c=1:size(CIRCULAR_INDEX,2)   
          r=CIRCULAR_R(c,:);
          p=S_tmp(CIRCULAR_INDEX(1,c),:);
          q=S_tmp(CIRCULAR_INDEX(2,c),:);
          e_p=E_tmp(CIRCULAR_INDEX(1,c),:);
          e_q=E_tmp(CIRCULAR_INDEX(2,c),:);
          E_tmp(CIRCULAR_INDEX(1,c),:)= (e_p .* q - e_q.*p).*q./r.^3;
          E_tmp(CIRCULAR_INDEX(2,c),:)= (e_q .* p - e_p.*q).*p./r.^3;
        end
      end

      n_error(sum(NET(1:i-1))+1:sum(NET(1:i)),:) = E_tmp;
    
    end  

% dw = -matrices2vector(dW); % old 
dw = matrices2vector(dW); % new <<<<<<<<<<<< 


if INVERSE
  if FIXED_WEIGHTS
    dw = reshape(E_tmp,numel(E_tmp),1);         % new <<<<<<<<<<<< 
    % dw = -reshape(E_tmp,prod(size(E_tmp)),1); % old 
    w=zeros( numel(train_in) , 1 ); % weight decay off 
  else
    dw = [ reshape(E_tmp,num_elements,1) ; dw ];   % new <<<<<<<<<<<< 
    %dw = [ -reshape(E_tmp,num_elements,1) ; dw ]; % old 
    if CIRCULAR  
      w=[zeros(num_elements,1);w]; % no weight decay for input      
    else
      w=[0.01*w_train_in;w]; % smooth (0.01) weight decay also for input values     % new <<<<<<<<<<<< 
    end
  end
end



dw = dw + WEIGHT_DECAY*w; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E,out]=error_hierarchic(w,train_in,train_out)
% [E,out]=error_hierarchic(w,train_in,train_out)

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NET    % Example:  NET=[   2  ,   4  ,   1  ,   4  ,   2  ]
global FCT    % Example:  FCT=['linr';'tanh';'linr';'tanh';'linr']
global WEIGHT_DECAY

global DATADIST
global WEIGHTED_DATA

global HIERARCHIC_MODE
global HIERARCHIC_VAR
global HIERARCHIC_LAYER
global HIERARCHIC_IDX

global INVERSE
global CIRCULAR

if INVERSE
   num_elements = NET(1)*size(train_out,2);
   train_in = reshape( w(1:num_elements) , NET(1) , size(train_out,2) );
   w_train_in=w(1:num_elements); % new <<<<<<<<<<<<<<<<<
   w   = w(num_elements+1 : end);
end

[net_dim,net_num] = size(NET);
[train_out_dim,P_num]  = size(train_out);
W=vector2matrices(w); 

hierarchic_idx = HIERARCHIC_IDX(:,HIERARCHIC_VAR~=0);
hierarchic_var = HIERARCHIC_VAR(HIERARCHIC_VAR~=0);
subnet_num = length(hierarchic_var);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out=zeros(train_out_dim,P_num,subnet_num);

S_bias = ones(1,P_num);  
%S_extract = train_in;
S_extract = feval( FCT(1,:) , train_in ); % in case of 'circ'

  for layer = 1:HIERARCHIC_LAYER-1
    S_extract = [S_bias;S_extract];
    S_extract = feval(FCT(layer+1,:),W{layer}*S_extract); 
  end

for subnet=1:subnet_num 
  S_recon = S_extract;
  S_recon(hierarchic_idx(:,subnet)==0,:)=zeros;

  for layer = HIERARCHIC_LAYER:net_num-1
    S_recon = [S_bias;S_recon];
    S_recon = feval(FCT(layer+1,:),W{layer}*S_recon); 
  end
 
  out(:,:,subnet) = S_recon;


  %%%%%%%%%%%%%%%%%%%%%%%
  % error function

  E_tmp=(S_recon - train_out).^2; 

  E_tmp(isnan(E_tmp))=0;  % NaN: missing data

  if WEIGHTED_DATA
    Eitemize(subnet)=0.5*sum(sum( DATADIST .* E_tmp ));
  else
    Eitemize(subnet)=0.5*sum(sum( E_tmp ));
  end

end


  E=hierarchic_var*Eitemize';
  
  % if WEIGHT_DECAY
    E = E + WEIGHT_DECAY*0.5*sum(w.^2); % weight decay for real weights
  % end
  
  % smooth (0.01) weight decay also for input values
  % but not in case of circular units
  if INVERSE && ~CIRCULAR, E = E + 0.01*WEIGHT_DECAY*0.5*sum(w_train_in.^2); end % new <<<<<<<<<<<< 
  
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dw_total,Etotal,n_error,n_out]=derror_hierarchic(w,train_in,train_out)
% gradient of error-fuction of hierarchic NLPCA
%
% [dw,E,n_error,n_out]=derror_hierarchic(w,train_in,train_out)
%
% dw - gradient (positive gradient, uphill direction)
% E  - error E(w)
%

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NET    % Example:  NET=[   2  ,   4  ,   1  ,   4  ,   2  ]
global FCT    % Example:  FCT=['linr';'tanh';'linr';'tanh';'linr']
global WEIGHT_DECAY 

global DATADIST
global WEIGHTED_DATA

global HIERARCHIC_VAR
global HIERARCHIC_LAYER
global HIERARCHIC_IDX

global INVERSE

global CIRCULAR
global CIRCULAR_INDEX
global CIRCULAR_R

if INVERSE
   num_elements = NET(1)*size(train_out,2);
   train_in = reshape( w(1:num_elements) , NET(1) , size(train_out,2) );
   w_train_in=w(1:num_elements); % new <<<<<<<<<<<<<<<<<
   w   = w(num_elements+1 : end);
end


[net_dim,net_num] = size(NET);
[train_in_dim,train_in_num] = size(train_in);
W=vector2matrices(w); 

subnet_num = length(HIERARCHIC_VAR);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Epattern=zeros([size(train_out),subnet_num]);
n_out=zeros(sum(NET),train_in_num,subnet_num);% n_out get the outputs of all units
for subnet=1:subnet_num, n_out(1:train_in_dim,:,subnet) = feval( FCT(1,:) , train_in ); end

if INVERSE
  for subnet=1:subnet_num, n_out(HIERARCHIC_IDX(:,subnet)==0,:,subnet)=zeros; end
end


%%%%%%%%%%%%%%%%%%%%%%%
% forward propagation                                  
  
  for subnet=1:subnet_num    
    if HIERARCHIC_VAR(subnet) ~= 0                 
      S_bias = ones(1,train_in_num);  
      for i = 1:net_num-1
     
        if i==1, n_begin = 1;  else n_begin=sum(NET(1:i-1))+1;  end
        S_in = [S_bias;n_out(n_begin:sum(NET(1:i)),:,subnet)];
    
        S_out =  feval(FCT(i+1,:),W{i}*S_in); 
 
        if i==(HIERARCHIC_LAYER-1)  
          S_out(HIERARCHIC_IDX(:,subnet)==0,:)=zeros; 
        end

        n_out(sum(NET(1:i))+1:sum(NET(1:i+1)),:,subnet) = S_out;      
      end 
      output = n_out(sum(NET(1:end-1))+1:end,:,subnet);          
      Epattern(:,:,subnet) = output - train_out;                   
    end
  end

%%%%%%%%%%%%%%%%%%%%%%%
% error function

  Epattern = Epattern.^2;
  
  Epattern(isnan(Epattern))=0;  % NaN: missing data

  if WEIGHTED_DATA
    for subnet=1:subnet_num
      Epattern(:,:,subnet)=Epattern(:,:,subnet).*DATADIST;
    end
  end

  Eitemize = 0.5*sum(sum(Epattern));
  Eitemize = reshape(Eitemize,[1,numel(Eitemize)]);

  Etotal=sum(HIERARCHIC_VAR.*Eitemize);

 
  % if WEIGHT_DECAY
    Etotal = Etotal + WEIGHT_DECAY*0.5*sum(w.^2); % weight decay for real weights
  % end  
  
  % smooth (0.01) weight decay also for input values
  % but not in case of circular units
  if INVERSE && ~CIRCULAR, Etotal = Etotal + 0.01*WEIGHT_DECAY*0.5*sum(w_train_in.^2); end % new <<<<<<<<<<<< 
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%< BACK PROPAGATION >%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% back propagation
  n_error = zeros(sum(NET),train_in_num,subnet);  % get the Errors of all neurons
  dW   = cell(1,net_num-1);           % Gradient
  W_bp = cell(1,net_num-1);           % special weight-matrix for backprob
                                      % Important in case 'total connected'  

% Weight-matrices preparing for back propagation  
      for u=1:net_num-1
        W_bp{u}=W{u}(:,2:NET(u)+1);  % cats the weights which belong to bias
      end  
 
dw=zeros(length(w),subnet);
for subnet=1:subnet_num  
  if HIERARCHIC_VAR(subnet) ~= 0  
    % last layer  
      S_tmp = n_out(end-NET(end)+1:end,:,subnet);        
      %if     FCT(end,:) == 'tanh'    E_tmp = (1-S_tmp.^2).*(train_out-S_tmp); % old
      %elseif FCT(end,:) == 'linr'    E_tmp = train_out-S_tmp;   end           % old
      if     FCT(end,:) == 'tanh',    E_tmp = (1-S_tmp.^2).*(S_tmp-train_out);  % new <<<<<<<<<<<<<<<<<
      elseif FCT(end,:) == 'linr',    E_tmp = S_tmp-train_out;   end            % new <<<<<<<<<<<<<<<<<
      E_tmp(isnan(E_tmp))=0;                      % NaN: missing data
      if WEIGHTED_DATA, E_tmp=E_tmp.*DATADIST; end % weighted data 
      n_error(end-NET(end)+1:end,:,subnet) = E_tmp;

    % all other layers
      for n=1:net_num-1
        i=net_num-n;  % counts backwards 
        %E_tmp = n_error(sum(NET(1:i))+1 : sum(NET(1:i+1)),:); % already exist
        S_tmp = n_out(sum(NET(1:i-1))+1:sum(NET(1:i)),:,subnet);
        
        if i==(HIERARCHIC_LAYER-1)  
          E_tmp(HIERARCHIC_IDX(:,subnet)==0,:)=zeros; 
        end

        dW{i} = E_tmp * [S_bias;S_tmp]';   % Gradient        
     
        if     FCT(i,:) == 'tanh',    E_tmp = (1-S_tmp.^2).*(W_bp{i}'*E_tmp);
        elseif FCT(i,:) == 'linr',    E_tmp = W_bp{i}'*E_tmp;   
        elseif FCT(i,:) == 'circ',    E_tmp = W_bp{i}'*E_tmp;   
          for c=1:size(CIRCULAR_INDEX,2)   
            r=CIRCULAR_R(c,:);
            p=S_tmp(CIRCULAR_INDEX(1,c),:);
            q=S_tmp(CIRCULAR_INDEX(2,c),:);
            e_p=E_tmp(CIRCULAR_INDEX(1,c),:);
            e_q=E_tmp(CIRCULAR_INDEX(2,c),:);
            E_tmp(CIRCULAR_INDEX(1,c),:)= (e_p .* q - e_q.*p).*q./r.^3;
            E_tmp(CIRCULAR_INDEX(2,c),:)= (e_q .* p - e_p.*q).*p./r.^3;
          end
        end
        
        n_error(sum(NET(1:i-1))+1:sum(NET(1:i)),:,subnet) = E_tmp;   
      end  
 
      % dw(:,subnet) = -matrices2vector(dW); % old
      dw(:,subnet) = matrices2vector(dW); % new <<<<<<<<<<<<<<<<<
  end
end


if INVERSE
  dw=[zeros(num_elements,subnet_num);dw]; 
  for subnet=1:subnet_num
    E_tmp = n_error(1:NET(1),:,subnet);
    E_tmp(HIERARCHIC_IDX(:,subnet)==0,:)=zeros;
    %dw(1:num_elements,subnet) = -reshape(E_tmp,num_elements,1); % old
    dw(1:num_elements,subnet) = reshape(E_tmp,num_elements,1);   % new <<<<<<<<<<<<<<<<<
  end
  if CIRCULAR  
    w=[zeros(num_elements,1);w]; % no weight decay for input      
  else
    w=[0.01*w_train_in;w]; % smooth (0.01) weight decay also for input values     % new <<<<<<<<<<<< 
  end
end


    dw_total=zeros(size(w));
    for subnet=1:subnet_num
      dw_total = dw_total + HIERARCHIC_VAR(subnet) * dw(:,subnet);
    end

    % if WEIGHT_DECAY  
      dw_total = dw_total + WEIGHT_DECAY*w; 
    % end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function w=matrices2vector(W)
% w=matrices2vector(W)
%
% W is a cell array of matrices
%
% matrices2vector returns a vector 
% whose elements are taken columwise
% from all matrices W{i}

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[W_dim,W_num] = size(W);

w=[];
for i=1:W_num
  w=[w;reshape( W{i} , prod(size(W{i})) , 1 )];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function W=vector2matrices(w)
% W=vector2matrices(w)
%
% w is a vector
%
% vector2matrice returns a cell array of matrices W{i}
%
% the sizes of matrices are taken from NET and CONNECT
% in NET are the number of neurons per layer [2,4,1,4,2]
% CONNECT == 1 means full-connected
% CONNECT == 2 means total-connected
%
% the elements are taken columwise from the vektor w

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NET

%global CONNECT

[net_dim,net_num] = size(NET);

pos_begin=1;
pos_end=0;
for i=1:net_num-1

%  if CONNECT == 1  % full-connected
    W_size=[NET(i+1),NET(i)+1];
%  end
  
%  if CONNECT == 2  % total-connected
%    W_size=[NET(i+1),sum(NET(1:i))+1];
%  end
  
  pos_end=pos_end+prod(W_size);
  W{i}=reshape(w(pos_begin:pos_end),W_size);
  pos_begin=pos_end+1;

end  

if pos_end < length(w)
  fprintf(2,'ERROR in vector2matrice -- w has to many elements\n');
  %size_w=size(w)
  %NET
  stop
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function w = set_weights_randomly(tmp)
% function w = set_weights_randomly

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global NET


  net_bias = NET+1;
  w_num=0;

 
  for i=2:length(NET)
    w_num=w_num + net_bias(i-1)*NET(i);
  end

  w=0.2*(rand(w_num,1)-0.5); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E_train,E_test]=get_error(w,error_function)
% get train and test error
%  [E_train,E_test]=get_error(w,error_func)
% 
% result is  MSE of train and test data without using weight-decay 
%  error = mean(mean(( net(data_in)-data_out ).^2)) 
% 
% WEIGHT_DECAY is set to zero (OFF)
% DATADIST is set to sign(DATADIST), NaN's are ignored but no weights are used
% error_func resp. 'h.error_function' are used as error function
% used data sets are TRAIN_IN TRAIN_OUT  TEST_IN TEST_OUT

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global SAVE_ERROR PRINT_ERROR
global TRAIN_IN TRAIN_OUT
global TEST_IN  TEST_OUT
global NET   
global WEIGHT_DECAY
global WEIGHTED_DATA DATADIST
global HIERARCHIC_LAYER
global HIERARCHIC_VAR
%global HIERARCHIC_IDX
global CLASSIFICATION

weight_decay_tmp = WEIGHT_DECAY;
datadist_tmp     = DATADIST;
WEIGHT_DECAY = 0;
DATADIST     = DATADIST > 0;


if HIERARCHIC_VAR
  %hierarchic_idx = HIERARCHIC_IDX(:,HIERARCHIC_VAR~=0);
  hierarchic_var = HIERARCHIC_VAR(HIERARCHIC_VAR~=0);
  subnet_num = length(hierarchic_var);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% propagation with traindata
  [E,reconstruct]= feval(error_function,w,TRAIN_IN,TRAIN_OUT);
  % older version: reconstruct = reconstruct(end-NET(end)+1:end,:,:); 
                                                       
  if CLASSIFICATION
    [m,out_idx]=max(reconstruct);
    [m,target_idx]=max(TRAIN_OUT);
  end


  [s1,s2,s3]=size(reconstruct);
  if s3==1 
    if CLASSIFICATION
      E_train=100-100*sum(out_idx==target_idx)/length(target_idx);
      missclass_train=sum(out_idx~=target_idx);
    else
      e = (reconstruct-TRAIN_OUT).^2;         
      E_train = mean(e(~isnan(e))); % NaN: missing data
      % E_train= mean(mean(e));
    end
  else
    E_train=zeros(1,subnet_num); 
    for subnet=1:subnet_num
      if CLASSIFICATION
        E_train(subnet)...
         =100-100*sum(out_idx(:,:,subnet)==target_idx)/length(target_idx);
      else
        e = (reconstruct(:,:,subnet)-TRAIN_OUT).^2;
        E_train(subnet) = mean(e(~isnan(e))); % NaN: missing data
        % E_train(subnet)=mean(mean(e);
      end
    end
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% propagation with testdata
  E_test=[];
  if ~isempty(TEST_IN)
    [E,reconstruct]= feval(error_function,w,TEST_IN,TEST_OUT);
    %reconstruct = out(end-NET(end)+1:end,:,:);

    if CLASSIFICATION
      [m,out_idx]=max(reconstruct);
      [m,target_idx]=max(TEST_OUT);
    end

    if s3==1 
      if CLASSIFICATION
        E_test=100-100*sum(out_idx==target_idx)/length(target_idx);
        missclass_test=sum(out_idx~=target_idx);
      else
        e = (reconstruct-TEST_OUT).^2;         
        E_test = mean(e(~isnan(e))); % NaN: missing data
        % E_test=mean(mean((reconstruct-TEST_OUT).^2));
      end
    else
      E_test=zeros(1,subnet_num);
      for subnet=1:subnet_num
        if CLASSIFICATION
          E_test(subnet)...
           =100-100*sum(out_idx(:,:,subnet)==target_idx)/length(target_idx); 
        else
          e = (reconstruct(:,:,subnet)-TEST_OUT).^2;
          E_test(subnet) = mean(e(~isnan(e))); % NaN: missing data
          % E_test(subnet)=mean(mean((reconstruct(:,:,subnet)-TEST_OUT).^2));
        end
      end
    end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if PRINT_ERROR & length(E_train)==1
  if isempty(TEST_IN) 
    if CLASSIFICATION fprintf(1,'E_train: %2.2f percent; missclassification: %i\n',E_train(end),missclass_train);  
    else fprintf(1,'E_train: %i\n',E_train(end)); end; 
  else
    if CLASSIFICATION
      fprintf(1,'E_train: %2.2f | E_test: %2.2f percent |  missclass: train: %i test: %i\n',E_train(end),E_test(end),missclass_train,missclass_test); 
    else
      fprintf(1,'E_train: %i  E_test: %i\n',E_train(end),E_test(end)); 
    end 
  end
end
if PRINT_ERROR & length(E_train)>1 
  Error=[E_train;E_test]  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WEIGHT_DECAY = weight_decay_tmp;
DATADIST     = datadist_tmp;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function h=check_semantic(h)
%
% redundancy check of network parameter in data struct 'network.xxx'
%

% Author   : Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global TRAIN_IN
global TRAIN_OUT

  if strcmp(h.weighted_data,'yes') && isempty(h.data_weight_out)
     error('Please specify: .data_weight_out')
  end

  if h.units_per_layer(end) ~= size(TRAIN_OUT,1)
    s1=num2str(h.units_per_layer(end));
    s2=num2str(size(TRAIN_OUT,1));
    error(['Dimension of output data ',s2,' ~= ',s1,' number of output units'])
  end
  if ~isempty(TRAIN_IN)
   if h.units_per_layer(1) ~= size(TRAIN_IN,1)
    s1=num2str(h.units_per_layer(1));
    s2=num2str(size(TRAIN_IN,1));
    error(['Dimension of input data ',s2,' ~= ',s1,' number of input units'])
   end
  end

  if h.units_per_layer(h.component_layer) ~= h.number_of_components
    s1='number of components and number of units in component layer '; 
    s2='must be the same: ';
    s3=num2str(h.number_of_components);
    s4=num2str(h.units_per_layer(h.component_layer));
    error([s1,s2,s3,' ~= ',s4])
  end

  if h.units_per_layer(h.component_layer) > size(TRAIN_OUT,1)
    s1=num2str(h.units_per_layer(h.component_layer));
    s2=num2str(size(TRAIN_OUT,1));
    error(['Number of components (',s1,') is bigger than data dimensionality (',s2,')'])
  end
  
  idx_NaN_out=isnan(h.data_train_out);
  num_NaN_out=sum(sum(idx_NaN_out));
  % if (num_NaN_out > 0) & strcmp(h.pre_pca,'yes') 
  %  error(['pca preprocessing does not work on missing data (NaN)'])
  % end

  if strcmp(h.pre_scaling,'no') && ~isempty(h.scaling_factor)
    fprintf(1,'\n# WARNING  Why is a scaling factor defined') 
    fprintf(1,' (%4.2f) and not used ? (scaling=no) \n\n',h.scaling_factor);
  end

  idx_NaN_in=isnan(h.data_train_in);
  num_NaN_in=sum(sum(idx_NaN_in));
  if (num_NaN_in > 0)
    error('NaN in input! (use data only as output and "type=inverse")')
  end

  if strcmp(h.type,'inverse') && ~isempty(h.data_train_in) 
   fprintf(1,'\n# WARNING  ".data_train_in" should be empty in inverse type\n')
  end

  if strcmp(h.type,'bottleneck') && (num_NaN_in == 0) && (num_NaN_out > 0)
    error('"type=bottleneck" and NaN only in ".data_train_out" does not work yet')
  end

  if ~isempty(h.data_train_in)
    if size(h.data_train_in,2)  ~= size(h.data_train_out,2)
       s1='data_in and data_out must have the same number of samples:';
       s2=num2str(size(h.data_train_in,2));
       s3=num2str(size(h.data_train_out,2));
       error([s1,s2,' ~= ',s3])
    end
  end

  if (length(h.units_per_layer)) ~= size(h.functions_per_layer,1)
     s1='.units_per_layer must be equal to .functions_per_layer:';
     s2=num2str(length(h.units_per_layer));
     s3=num2str(size(h.functions_per_layer,1));
     error([s1,' ',s2,' ~= ',s3])
  end
%  if (length(h.units_per_layer)-1) ~= size(h.functions_per_layer,1)
%     s1='.units_per_layer-1 must be equal to .functions_per_layer:';
%     s2=num2str(length(h.units_per_layer));
%     s3=num2str(size(h.functions_per_layer,1));
%     error([s1,' ',s2,'-1 ~= ',s3])
%  end

  if (h.weight_decay_coefficient < 0) % | (h.weight_decay_coefficient > 1) 
    error('''.weight_decay_coefficient'' must be positive\n')
  elseif (h.weight_decay_coefficient > 1) 
   s='\n# WARNING ''.weight_decay_coefficient'' should be between 0 and 1\n\n';
   fprintf(1,s) 
  end

  if strcmp(h.mode,'hierarchic')
  if length(h.hierarchic_coefficients)~=h.units_per_layer(h.hierarchic_layer)+1
     s0='<length(HIERARCHIC_VAR)=';
     s1=num2str(length(h.hierarchic_coefficients));
     s2=num2str(h.units_per_layer(h.hierarchic_layer));
     error([s0,s1,'> ~= <number hierarchic_layer=',s2,'  + 1>'])
   end
  end

  % h.sort_components
  if ( strcmp(h.mode,'hierarchic')  && ... 
       strcmp(h.sort_components,'no') && h.number_of_components>1 ),
     fprintf(1,'\n# WARNING Why not sorting components ?? Hierarchic mode!!\n\n')
  end
  if ( strcmp(h.mode,'symmetric')  && strcmp(h.sort_components,'yes') ),
     fprintf(1,'\n# WARNING Sorting components in symmetric mode. Why ??\n\n');
  end

  if strcmp(h.weight_initialization,'set_weights_linear') ... 
    && strcmp(h.pre_pca,'no') % |  h.number_of_components~=size(TRAIN_OUT,1) ),
    fprintf(1,'\n# WARNING weight_initialization: ''set_weights_randomly''')
    fprintf(1,'would be a better choise\n')
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

