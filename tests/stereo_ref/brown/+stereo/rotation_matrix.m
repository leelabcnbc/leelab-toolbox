function [rotationMat] = rotation_matrix(theta, axisIndex)
% ROTATION_MATRIX Create rotation matrix for rotation around axis_index 
%  
%   standard rotation matrix used in CG, axis_index is 1 (x), 2 (y), 3 (z).
%   theta is given in radians.
%   
%   reference: 11.2.1 of Computer Graphics: Principles and Practice 3rd ed.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 15-Feb-2014 03:29:37 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : rotation_matrix.m 

assert(axisIndex >= 1 && axisIndex <= 3);

axisNameArray = {'xrotate','yrotate','zrotate'};
axisName = axisNameArray{axisIndex};

rotationMat = makehgtform(axisName,theta);
rotationMat = rotationMat(1:3,1:3); % the trailing 1 is not needed.

end



% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [rotation_matrix.m] ======  
