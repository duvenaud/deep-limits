function [S,dz] = nlpca_get_data(net,pc)
% data = nlpca_get_data(net,pc)  % generating data from new component values 'pc'
% data = nlpca_get_data(net)     % reconstruction of train data
% [data,dz] = nlpca_get_data(net,pc) % get data and drivatives for component values 'pc'
%
% using network architecture 'net'
% from [pc, net, network] = nlpca(data,k)
%
% pc   - component values z (scores)
%         row-wise are the individual components
%         columns represent the samples
% data - rows are the original variables (attributes)
%         columns are the samples
% dz   - derivative dx/dz of data x with respect 
%         to first component z (dz describes the tangent direction at the 
%         curve given by the first component for the sample point(s) defined in pc)
%
% See also: nlpca_get_components nlpca nlpca_plot
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
global CIRCULAR_INDEX

if strcmp(net.circular,'yes')
    cidx=net.circular_idx;
    pos=find(cidx==1);
    for i=1:size(pos,2)
      cidx=cidx([1:pos(i)+(i-1),pos(i)+(i-1):end]);
    end
    cnidx=find(cidx==1);
    CIRCULAR_INDEX=reshape(cnidx,2,size(pos,2));
end

if ~isfield(net,'weight_matrices') 
  net=reduce_parameters(net);
end

fct=net.functions_per_layer;
 if (length(net.units_per_layer)-1) == size(net.functions_per_layer,1)
   fct=['linr';fct] % in case of older versions
 end
W=net.weight_matrices;


% propagation of component values through part II of the network
  if nargin==2 
    num_units=net.units_per_layer(net.component_layer);
    if size(pc,1) > num_units
     s1=['number of components (',num2str(size(pc,1)),') have to be equal '];
     s2=['or smaller than the number of '];
     error([s1,s2,'units in component layer (',num2str(num_units),')'])
    end
    if size(pc,1) < num_units
     s1=['# WARNING: number of components (',num2str(size(pc,1)),') is smaller'];
     s2=[' than \n#          the number of units in component layer'];
     s3=[' (',num2str(num_units),')\n'];
     s4=['#          -> residual components are set to zero!\n'];
     fprintf(1,[s1,s2,s3,s4])
    end

    %S=pc;
    S=zeros(num_units,size(pc,2));
    S(1:size(pc,1),:)=pc;
      if strcmp(net.circular,'yes') 
        S=net_phi2pq(S); % enlarge component layer
      end
    S=feval(fct(1,:),S);
    S_bias=ones(1,size(S,2));

    for i=net.component_layer : length(net.units_per_layer)-1
      S=[S_bias;S];
      S=feval(fct(i+1,:),W{i}*S);
    end
  end


% propagation of input data
  if nargin==1 
    S=net.data_train_in;
      if strcmp(net.circular,'yes') && strcmp(net.type,'inverse')
        S=net_phi2pq(S); %enlarge component (input) layer
      end; 
    S=feval(fct(1,:),S);
    S_bias=ones(1,size(S,2));

    for i= 1 : length(net.units_per_layer)-1
      S=[S_bias;S];
      S=feval(fct(i+1,:),W{i}*S);
    end
  end

  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get derivative with respect to first component

if nargout==2 % calcuation of derivative only if requested

  if nargin==1 % used original components, if pc is not given as input
    pc=nlpca_get_components(net);
  end    

  if size(pc,1) > 1
    pc=pc(1,:); % use only first component  
    fprintf(1,'# derivative ''dz'' is calculated with respect to ''first'' component only\n'); 
  end


  % define derivative input dz and data input S
      num_units=net.units_per_layer(net.component_layer);
      S=zeros(num_units,size(pc,2));
      dz=zeros(num_units,size(pc,2));

    if strcmp(net.circular,'yes') && net.circular_idx(1)==1,
      S(1,:) = cos(pc);
      S(2,:) = sin(pc);
      dz(1,:) = -sin(pc);
      dz(2,:) =  cos(pc);
    else
      S(1,:) = pc;
      dz(1,:) = ones(size(pc));
    end

    
  % propagate dz through the network (part II)
        S_bias  = ones(1,size(S,2));
        dz_bias = zeros(1,size(dz,2));
        
      for i=net.component_layer : length(net.units_per_layer)-1
        S=[S_bias;S];
        S=feval(fct(i+1,:),W{i}*S);
        dz=[dz_bias;dz];
        if     fct(i+1,:) == 'tanh',  dz=(1-S.^2).*(W{i}*dz);
        elseif fct(i+1,:) == 'linr',  dz=W{i}*dz;
        end
      end

end

 
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

