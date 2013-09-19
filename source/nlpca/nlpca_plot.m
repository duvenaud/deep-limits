function nlpca_plot(w_all,error_func,train_in,train_out)
% plot components of nonlinear PCA (NLPCA)
%
% nlpca_plot(net)     plot of original data and components 
%                      3d-plot by using the first 3 variables
% nlpca_plot(network) includes data scaling and transformations (pre-processing)
%                      3d-plot by using the first 3 linear PCs if standard 
%                      PCA is used for pre-processing
% 
% See also: nlpca
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% nlpca_plot(w,error_func,train_in,train_out) % while training
% 'error_func.m' is used as propagation function
% WEIGHT_DECAY is set to zero (OFF)
% DATADIST is set empty [] , not considered  (grid data set)

global GROUP
global NET   
global FCT    % Example:  FCT=['linr';'tanh';'linr';'tanh';'linr']
global WEIGHT_DECAY
global WEIGHTED_DATA DATADIST 
global COMPONENT_LAYER
global CIRCULAR
global CIRCULAR_IDX
global CIRCULAR_INDEX
global INVERSE

weight_decay_tmp = WEIGHT_DECAY;
weighted_data_tmp= WEIGHTED_DATA;
datadist_tmp     = DATADIST;
WEIGHT_DECAY = 0;
WEIGHTED_DATA= 0;
DATADIST     =[];

if nargin==4
   if INVERSE
     num_elements = NET(1)*size(train_out,2);
     train_in = reshape( w_all(1:num_elements) , NET(1) , size(train_out,2) );
     w        = w_all(num_elements+1 : end);
   else
     w = w_all;
   end
end


if ~(nargin==4)
  if nargin==1, hh=w_all; end

  NET= hh.units_per_layer;
  FCT=hh.functions_per_layer;
  if (length(hh.units_per_layer)-1) == size(hh.functions_per_layer,1)
    FCT=['linr';FCT]; % in case of older versions
  end
  COMPONENT_LAYER=hh.component_layer;
  train_in  = hh.data_train_in;
  train_out = hh.data_train_out;
  INVERSE   = strcmp(hh.type,'inverse');
  CIRCULAR  = strcmp(hh.circular,'yes');
  if CIRCULAR
    % set CIRCULAR_INDEX
      cidx=hh.circular_idx;
      pos=find(cidx==1);
      for i=1:size(pos,2)
        cidx=cidx([1:pos(i)+(i-1),pos(i)+(i-1):end]);
      end
      cnidx=find(cidx==1);
      CIRCULAR_INDEX = reshape(cnidx,2,size(pos,2));
      CIRCULAR_IDX   = hh.circular_idx;
    NET(hh.component_layer)=NET(hh.component_layer)+sum(hh.circular_idx);
    train_in  =net_phi2pq(train_in);
  end  
  if ~isempty(hh.data_class) % labels
    GROUP=hh.data_class;
  elseif hh.version >= 0.71, % let GROUP for older versions
    GROUP=[];
  end
  
  if isfield(hh,'weight_matrices') % reduced parameter struct
    w=matrices2vector(hh.weight_matrices);
  else % complete parameter struct 
    w=hh.weight_vector;
    % train_out pre-processing
    if strcmp(hh.pre_unit_variance,'yes')
        train_out=(train_out - repmat(hh.removed_mean,1,size(train_out,2)) )...
                   ./ repmat(hh.removed_std,1,size(train_out,2));
    end
    if strcmp(hh.pre_pca,'yes')
      if ~isempty(hh.pca_components)  % in case of NaN's in train_out
        train_out=hh.pca_components;  % using complete components
      else
        train_out=hh.eigenvectors * ...
                 (train_out - repmat(hh.pca_removed_mean,1,size(train_out,2)));
      end
    end 
    if strcmp(hh.pre_scaling,'yes')
        train_out=hh.scaling_factor*train_out;
    end
    if strcmp(hh.type,'bottleneck')
      % train_in pre-processing
      if strcmp(hh.pre_unit_variance,'yes')
          train_in=(train_in - repmat(hh.removed_mean,1,size(train_in,2)))...
                     ./ repmat(hh.removed_std,1,size(train_in,2));
      end
      if strcmp(hh.pre_pca,'yes')
          train_in=hh.eigenvectors * ... 
                  (train_in - repmat(hh.pca_removed_mean,1,size(train_in,2)));
      end 
      if strcmp(hh.pre_scaling,'yes')
          train_in=hh.scaling_factor*train_in;
      end
    end
  end
end

  error_func='error_func';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% propagation with traindata
  [E,out]= feval(error_func,w,train_in,train_out);
  reconstruct = out(end-NET(end)+1:end,:);
  E_tmp=(reconstruct-train_out).^2;
  E_tmp(isnan(E_tmp))=0;
  E=mean(mean( E_tmp ));

% component space (bottlenecklayer)
  out_bneck=out(sum(NET(1:COMPONENT_LAYER-1))+1:sum(NET(1:COMPONENT_LAYER)),:);

  if CIRCULAR
    [out_bneck]=net_pq2phi(out_bneck); 
  end
  bneck_num = size(out_bneck,1); 

% create input-data for bottlenecklayer
  if bneck_num == 1
   in_bneck=min(out_bneck):(max(out_bneck)-min(out_bneck))/1000:max(out_bneck);
   if CIRCULAR 
     in_bneck = -pi: 2*pi/30 : pi;
     in_bneck = net_phi2pq(in_bneck); 
   end
  end

  if bneck_num == 2
    x = min(out_bneck(1,:)) : (max(out_bneck(1,:))-min(out_bneck(1,:)))/30 : max(out_bneck(1,:));
    y = min(out_bneck(2,:)) : (max(out_bneck(2,:))-min(out_bneck(2,:)))/30 : max(out_bneck(2,:));
    if CIRCULAR 
       if CIRCULAR_IDX(1), x = -pi: 2*pi/30 : pi; end 
       if CIRCULAR_IDX(2), y = -pi: 2*pi/30 : pi; end        
    end
    [X,Y] = meshgrid(x,y);
    x_in=reshape( X , numel(X) , 1 )';
    y_in=reshape( Y , numel(Y) , 1 )';
    in_bneck = [x_in;y_in];
    if CIRCULAR 
      in_bneck = net_phi2pq(in_bneck); 
    end
    [tmp,zeroline_idx]=min(abs([x;y])');
  end

  if bneck_num >= 3
    x = min(out_bneck(1,:)) : (max(out_bneck(1,:))-min(out_bneck(1,:)))/30 : max(out_bneck(1,:));
    y = min(out_bneck(2,:)) : (max(out_bneck(2,:))-min(out_bneck(2,:)))/30 : max(out_bneck(2,:));
    z = min(out_bneck(3,:)) : (max(out_bneck(3,:))-min(out_bneck(3,:)))/30 : max(out_bneck(3,:));
    if CIRCULAR
      if CIRCULAR_IDX(1), x = -pi: 2*pi/30 : pi; end 
      if CIRCULAR_IDX(2), y = -pi: 2*pi/30 : pi; end  
      if CIRCULAR_IDX(3), z = -pi: 2*pi/30 : pi; end  
    end
    [X,Y] = meshgrid(x,y);
    x_in=reshape( X , prod(size(X)) , 1 )';
    y_in=reshape( Y , prod(size(Y)) , 1 )';
    z_in=zeros(1,length(x_in));
    in_bneck_xy = [x_in;y_in;z_in];
    [X,Z] = meshgrid(x,z);
    x_in=reshape( X , prod(size(X)) , 1 )';
    y_in=zeros(1,length(x_in));
    z_in=reshape( Z , prod(size(Z)) , 1 )';
    in_bneck_xz = [x_in;y_in;z_in];
    [Y,Z] = meshgrid(y,z);
    x_in=zeros(1,length(x_in));
    y_in=reshape( Y , prod(size(Y)) , 1 )';
    z_in=reshape( Z , prod(size(Z)) , 1 )';
    in_bneck_yz = [x_in;y_in;z_in];
    [tmp,zeroline_idx]=min(abs([x;y])');
    if CIRCULAR
      in_bneck_xy=net_phi2pq(in_bneck_xy); 
      in_bneck_xz=net_phi2pq(in_bneck_xz); 
      in_bneck_yz=net_phi2pq(in_bneck_yz); 
    end
  end



% propagation in reconstruction-part of the net (Part II of the net)
  W=vector2matrices(w);
  net_tmp = NET;
  fct_tmp = FCT;
  NET=NET(COMPONENT_LAYER:end);
  FCT=FCT(COMPONENT_LAYER:end,:);
  w=matrices2vector({W{COMPONENT_LAYER:end}});

  if bneck_num >= 3 
    P_out = zeros(NET(end),length(in_bneck_xy));
    [E,out_grid]= feval(error_func,w,in_bneck_xy,P_out);
    out_grid_xy = out_grid(end-NET(end)+1:end,:);
    [E,out_grid]= feval(error_func,w,in_bneck_xz,P_out);
    out_grid_xz = out_grid(end-NET(end)+1:end,:);
    [E,out_grid]= feval(error_func,w,in_bneck_yz,P_out);
    out_grid_yz = out_grid(end-NET(end)+1:end,:);
  else
    P_out = zeros(NET(end),length(in_bneck));
    [E,out_grid]= feval(error_func,w,in_bneck,P_out);
    out_grid = out_grid(end-NET(end)+1:end,:);
  end

NET = net_tmp;
FCT = fct_tmp;
WEIGHT_DECAY = weight_decay_tmp;
WEIGHTED_DATA= weighted_data_tmp;
DATADIST     = datadist_tmp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%  Plot 1 --> 2  %%%%%%%
   if bneck_num==1 && NET(end)==2
     plotc(  train_out(1,:),  train_out(2,:),'b.'); hold on;  a=axis;
     plotc(reconstruct(1,:),reconstruct(2,:),'ro'); hold on;  axis(a); 
     plot(out_grid(1,:),out_grid(2,:),'r-');        hold off; axis(a);
     axis off
   end 

%%%%%%%  Plot 2 --> 2  %%%%%%%
   if bneck_num==2 && NET(end)==2

     plotc(  train_out(1,:),  train_out(2,:),'b.'); hold on;
     plotc(reconstruct(1,:),reconstruct(2,:),'ro'); hold on; 
     X_out = reshape(out_grid(1,:) , size(X) );
     Y_out = reshape(out_grid(2,:) , size(X) );
     mymesh(X_out,Y_out,zeroline_idx); hold off;
     axis off
   end 
    
%%%%%%%  Plot 1 --> 3  %%%%%%%
   if bneck_num==1 && NET(end)>=3
     plot3c(  train_out(1,:),  train_out(2,:),  train_out(3,:),'b.'); hold on;
     plot3c(reconstruct(1,:),reconstruct(2,:),reconstruct(3,:),'ro'); hold on;
     plot3(out_grid(1,:),out_grid(2,:),out_grid(3,:),'r-'); hold off;
     axis('equal')
     axis off
   end 

%%%%%%%  Plot 2 --> 3  %%%%%%%
   if bneck_num==2 && NET(end)>=3
     plot3c(train_out(1,:),train_out(2,:),train_out(3,:),'.'); hold on;
     hold on
     x = reshape(out_grid(1,:) , size(X) );
     y = reshape(out_grid(2,:) , size(X) );
     z = reshape(out_grid(3,:) , size(X) );
     axis('equal')
     mesh(x,y,z)
     hold off 
     axis off  
   end 

%%%%%%%  Plot 3 --> 3  %%%%%%%
   if bneck_num>=3 && NET(end)>=3
     plot3c(train_out(1,:),train_out(2,:),train_out(3,:),'.'); hold on;
     axis('equal')
     hold on

     x = reshape(out_grid_xy(1,:) , size(X) );
     y = reshape(out_grid_xy(2,:) , size(X) );
     z = reshape(out_grid_xy(3,:) , size(X) );
     mesh(x,y,z)
 
     x = reshape(out_grid_xz(1,:) , size(X) );
     y = reshape(out_grid_xz(2,:) , size(X) );
     z = reshape(out_grid_xz(3,:) , size(X) );
     mesh(x,y,z)
  
     x = reshape(out_grid_yz(1,:) , size(X) );
     y = reshape(out_grid_yz(2,:) , size(X) );
     z = reshape(out_grid_yz(3,:) , size(X) );
     mesh(x,y,z)

     hold off  
     axis off
   end 

%%%%%%%  Plot 3 --> 2  %%%%%%%
   if bneck_num > NET(end)  &&  NET(end)==2
     plotc(  train_out(1,:),  train_out(2,:),'b.'); hold on;
     plotc(reconstruct(1,:),reconstruct(2,:),'ko'); hold on;  

     x = reshape(out_grid_xy(1,:) , size(X) );
     y = reshape(out_grid_xy(2,:) , size(X) );
     mymesh(x,y,zeroline_idx); hold off;

     axis off
   end 
  
   
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mymesh(X,Y,zeroline_idx)
%
% mymesh(X,Y) plots the 2-D-mesh defined by two matrix arguments.
%

% Author: Matthias Scholz

if ~(sum(size(X)==size(Y)) == 2)
  fprintf(2,'Error in mymesh -- matrices must have equal size\n');
end

[dim,num] = size(X);

hold on
  for i=1:num
    plot(X(:,i),Y(:,i),'r-')
  end
  for i=1:dim
    plot(X(i,:),Y(i,:),'r-')
  end
  plot(X(zeroline_idx(2),:),Y(zeroline_idx(2),:),'k-','LineWidth',2)
  plot(X(:,zeroline_idx(1)),Y(:,zeroline_idx(1)),'k-','LineWidth',2)
hold off
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

function plotc(x,y,s,ms)
% plots data with different colours depending on global variable GROUP
%
% plotc(x,y,'.',ms)  % ms - markersize
% plotc(data(1,:), data(2,:), '.',15)
%
% global GROUP; GROUP=[1,1,1,2,2,2,3,3,3,...]
%   1 - blue 
%   2 - red 
%   3 - green 
%   4 - magenta
%   5 - cyan 
%   6 - black 
%   7 - yellow
%   8 - white  
%
% set(gca,'XTickLabel',[])
% set(gca,'YTickLabel',[]) 
%
% print2file('fig_test',1.2)
%
% print -depsc fig_test.eps
% print -deps  fig_test_bw.eps
% print -dpng  fig_test.png
%
% a=axis;a(1)=xx;a(2)=xx;axis(a);

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global GROUP 

if isempty(GROUP) 
  plot(x,y,s)
else
    
  if nargin<=3
    ms=6; % markersize
  end

  if size(GROUP,2) ~= size(x,2)
  fprintf(...
  '# WARNING <plotc>: number of labels in GROUP ~= number of samples\n');
  end


  if nargin==1
    [dim,num]=size(x);

    for i=1:dim
      plotc(x(i,:),i*ones(1,num),'.'); hold on
    end
    a=axis;a(3)=0;a(4)=dim+1;axis(a);
    hold off

  else

    if length(s)==2,  s=s(2:end); end
    plot(x(GROUP==1),y(GROUP==1),['b',s],'markersize',ms); hold on
    plot(x(GROUP==2),y(GROUP==2),['r',s],'markersize',ms);
    plot(x(GROUP==3),y(GROUP==3),['g',s],'markersize',ms);
    plot(x(GROUP==4),y(GROUP==4),['m',s],'markersize',ms);
    plot(x(GROUP==5),y(GROUP==5),['c',s],'markersize',ms);
    plot(x(GROUP==6),y(GROUP==6),['k',s],'markersize',ms);
    plot(x(GROUP==7),y(GROUP==7),['y',s],'markersize',ms); 
    %plot(x(GROUP==8),y(GROUP==8),['w',s],'markersize',ms); 
    hold off; 

  end
end

set_axis([x;y],0.07,0) % arrange the axis by the data, not by the component


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot3c(x,y,z,s)
% plot3c(x,y,z,'.')
%
% plots data with different colours depending on global variable GROUP
%
% 1 - blue 
% 2 - red 
% 3 - green 
% 4 - magenta
% 5 - cyan 
% 6 - black 
% 7 - yellow
% 8 - white  
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global GROUP 

if isempty(GROUP) 
  plot3(x,y,z,s)
else
  if length(s)==2,  s=s(end); end
  plot3(x(GROUP==1),y(GROUP==1),z(GROUP==1),['b',s]); hold on
  plot3(x(GROUP==2),y(GROUP==2),z(GROUP==2),['r',s]);
  plot3(x(GROUP==3),y(GROUP==3),z(GROUP==3),['g',s]);
  plot3(x(GROUP==4),y(GROUP==4),z(GROUP==4),['m',s]);
  plot3(x(GROUP==5),y(GROUP==5),z(GROUP==5),['c',s]);
  plot3(x(GROUP==6),y(GROUP==6),z(GROUP==6),['k',s]);
  plot3(x(GROUP==7),y(GROUP==7),z(GROUP==7),['y',s]); 
  plot3(x(GROUP==8),y(GROUP==8),z(GROUP==8),['w',s]); hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function set_axis(fs,r,opt)
% set_axis(fs,r,opt)
% set_axis(fs,0.04,1)
%
% plot the data with distance r to the axis
%
% default r=0.04

% opt == 1 : set(gca,'XTickLabel',[])
%            set(gca,'YTickLabel',[])
%

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 1; r=0.04; end
if nargin < 3; opt=0; end

if ~isempty(fs)
  fs_min=min(fs,[],2);
  fs_max=max(fs,[],2);
  fs_dist=fs_max-fs_min;
  fs_dist(fs_dist==0)=1;

  axis([fs_min(1)-r*fs_dist(1),...
        fs_max(1)+r*fs_dist(1),...
        fs_min(2)-r*fs_dist(2),...
        fs_max(2)+r*fs_dist(2)])
end

if opt==1
 set(gca,'XTickLabel',[]);
 set(gca,'YTickLabel',[]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
