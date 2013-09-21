function c = colorbrew( i )
%
% Nice colors taken from 
% http://colorbrewer2.org/
%
% David Duvenaud
% March 2012

c_array(1, :) = [ 222, 45, 48 ];   % red
c_array(2, :) = [ 94, 80, 224];  % blue
c_array(3, :) = [ 47, 175, 44 ];   % green
c_array(4, :) = [ 1, 80, 180 ];  % purple
c_array(5, :) = [ 255, 127, 50 ];   % orange
c_array(6, :) = [ 237, 248, 177 ];  % yellow
c_array(7, :) = [ 166, 86, 40 ];   % brown
c_array(8, :) = [ 247, 129, 191 ]; % pink
c_array(9, :) = [ 153, 153, 153];  % grey
c_array(10, :) = [ 0, 0, 0];       % black
c_array(11, :) = [158, 202, 225];  % light blue
c_array(12, :) = [222, 235, 247];  % really light blue
c_array(13, :) = [173, 221, 142];  % yellow-green


c = c_array( mod(i - 1, length(c_array)) + 1, : ) ./ 255;
end
