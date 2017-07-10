function [aerMatrix] = cart2retina2(xyzMatrix)
% CART2RETINA2 cart system to retina2 system 
%  
%   See README for detail. 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 13-Jul-2014 09:28:32 $ 
%% DEVELOPED : 8.3.0.532 (R2014a) 
%% FILENAME  : cart2retina2.m 

% assume a 3 by N matrix.
assert(size(xyzMatrix,1)==3 && ismatrix(xyzMatrix));

aerMatrix = zeros(size(xyzMatrix));

% fill in azi, 1st col
aerMatrix(1,:) = atan2(-xyzMatrix(1,:),-xyzMatrix(3,:));
% fill in elev, 2nd col
aerMatrix(2,:) = atan2(xyzMatrix(2,:), sqrt( xyzMatrix(3,:).^2 + xyzMatrix(1,:).^2 ) );
% fill in r, 3rd col
aerMatrix(3,:) = sqrt(sum( xyzMatrix.^2,1)); 

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [cart2retina2.m] ======  
