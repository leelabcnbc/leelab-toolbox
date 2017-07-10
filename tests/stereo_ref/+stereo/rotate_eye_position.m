function [L,R, RyF] = rotate_eye_position(F,IODistance)
% ROTATE_EYE_POSITION ... 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 07-Apr-2014 15:34:45 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : rotate_eye_position.m 

if nargin < 2
    IODistance = 0.065;
end

L = [-IODistance/2;0;0];
R = [IODistance/2;0;0];

Fretina2 = stereo.cart2retina2(F); % convert fixation point in retina2 coordinate.
[RyF] = stereo.rotation_matrix(Fretina2(1), 2); % compute rotation matrix for head.

% 
% anglesLR = atan2( -F(1),-F(3)); 
% [RyF] = stereo.rotation_matrix(anglesLR, 2);

R = RyF*R;
L = RyF*L;

% tempLR = F-R;
% 
% anglesLR = atan2( -tempLR(1),-tempLR(3)); 
% [RyF] = stereo.rotation_matrix(anglesLR, 2);
% 
% R = RyF*R;
% 
% 
% tempLR = F-L;
% 
% anglesLR = atan2( -tempLR(1),-tempLR(3)); 
% [RyF] = stereo.rotation_matrix(anglesLR, 2);
% 
% L = RyF*L;

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [rotate_eye_position.m] ======  
