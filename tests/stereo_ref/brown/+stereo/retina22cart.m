function xyzMatrix = retina22cart(aerMatrix)
% RETINA22CART retina2 system to cart system 
%  
%   see README for detail.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 13-Jul-2014 09:54:46 $ 
%% DEVELOPED : 8.3.0.532 (R2014a) 
%% FILENAME  : retina22cart.m 

% assume a 3 by N matrix.
assert(size(aerMatrix,1)==3 && ismatrix(aerMatrix));
xyzMatrix = zeros(size(aerMatrix));


xyzMatrix(2,:) = aerMatrix(3,:) .* sin( aerMatrix(2,:) );
rcoselev = aerMatrix(3,:) .* cos( aerMatrix(2,:) );
xyzMatrix(3,:) = -rcoselev .* cos( aerMatrix(1,:) );
xyzMatrix(1,:) = -rcoselev .* sin( aerMatrix(1,:) );

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [retina22cart.m] ======  
