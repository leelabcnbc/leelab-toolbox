function [R,T] =  transformation_fixation(F, E)
% TRANSFORMATION_FIXATION ... 
%  
%   Given fixation coordinate F (fixation), and eye coordinate E (eye),
%   compute the translation vector T and rotation matrix R so that F' =
%   R*(F-T) is along z-axis, in the eye coordinate system.
%   
%   basically some copy from cortex_toolkit.
%   
%   E better has the form (X,0,0). Other forms may work as well, but not
%   guaranteed.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 05-Apr-2014 21:29:26 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : transformation_fixation.m 

F = F(:);
assert(numel(F)==3);
% for safety.
% assert(F(3) < 0);

if nargin < 2 || isempty(E)
    E = [0;0;0]; % default E is origin.
end

assert(numel(E)==3);
% for safety.
% assert(all(E(2:3)==0));
assert((E(2)==0)); % only allow eye in the XoZ plane.
T = E; % usually so trivial... here just for completeness.

% angles = rotation_angles_inner(F, E);

% compute the retinal position of fixation point in eye coordinate, after the eye
% cooridinate is just a translation from the world coordiante.

angles = stereo.cart2retina2(F-E); % the coordinate of F after translation is F-E, because we need to make E in new system being 0.

% compute rotation matrix (reverse) to make F now have 0 azimuth and 0
% elevation.
[RyF] = stereo.rotation_matrix(-angles(1), 2); % along Y
[RxF] = stereo.rotation_matrix(-angles(2), 1); % along X.
           
R = RxF*RyF; % we first rotate along Y axis, then along X axis. 

end

% function anglesLR = rotation_angles_inner(F, R)
% 
% anglesLR = zeros(2,1);
% 
% tempLR = F-R;
% % anglesLR(1) = atan(tempLR(1)/tempLR(3)); % must be sth in the range of 
% % % eq. used in the paper, which is correct.
% % anglesLR(2) = atan( tempLR(2)/  ( -sin(anglesLR(1))*tempLR(1) - cos( anglesLR(1) ) * tempLR(3)   )   );
% 
% % for every angle
% % anglesLR = zeros(2,1);
% % actually, we want to calculate the angle of point F-R, w.r.t. minus z
% % axis. so we add minus before.
% anglesLR(1) = atan2( -tempLR(1),-tempLR(3)); 
% anglesLR(2) = atan2( tempLR(2), sqrt(tempLR(1).^2 + tempLR(3).^2));
% end






% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [transformation_fixation.m] ======  
