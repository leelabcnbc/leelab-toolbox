function disparityMap =  cart2disparity(imdata_reshaped, fixationThis, infinite_fixation, IODistance)
% CART2DISPARITY convert cartesian coordinates to disparity. 
%  
%   3xN matrix to Nx1 disparity vector. 
%   a rewrite of cart2disparity_old. They should perform equally well,
%   since they should only differ in using cart2retina or cart2retina2.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 13-Jul-2014 12:22:09 $ 
%% DEVELOPED : 8.3.0.532 (R2014a) 
%% FILENAME  : cart2disparity.m 

if nargin < 3 || isempty(infinite_fixation)
    infinite_fixation = false;
end

if nargin < 4 || isempty(IODistance)
    IODistance = 0.065; % human inter ocular distance.
end

assert(ismatrix(imdata_reshaped));
assert(size(imdata_reshaped,1)==3);

assert(numel(fixationThis) == 3); % must contain 3 elements.
fixationThis = fixationThis(:); % for convenience, I here coerce it to column vector.

% given the fixation point and IO distance, compute the positions of the
% centers of two eyes, in the world XYZ coordinate sytem.
[L,R] = stereo.rotate_eye_position(fixationThis,IODistance);

% compute rotation matrices to make fixation point on -Z axis.
if ~infinite_fixation
    RotationL = stereo.transformation_fixation(fixationThis,L);
    RotationR = stereo.transformation_fixation(fixationThis,R);
else
    RotationL = stereo.transformation_fixation(fixationThis,[0;0;0]);
    RotationR = RotationL;
end

xyzMatrixL = RotationL*bsxfun(@minus,imdata_reshaped,L);
xyzMatrixR = RotationR*bsxfun(@minus,imdata_reshaped,R);

PLiRetina2 = stereo.cart2retina2(xyzMatrixL);
PRiRetina2 = stereo.cart2retina2(xyzMatrixR);

disparityMap = PLiRetina2(1,:)-PRiRetina2(1,:);

% fix wrap around.
disparityMap(disparityMap > pi) = disparityMap(disparityMap > pi) - 2*pi;
disparityMap(disparityMap < -pi) = disparityMap(disparityMap < -pi) + 2*pi;


end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [cart2disparity.m] ======  
