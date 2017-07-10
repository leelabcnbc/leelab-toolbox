function [xyzMatrix,noResultMask,aziShift] = cmu_raw_to_xyz_centered_new(datamap)
% CMU_RAW_TO_XYZ_CENTERED_NEW ... 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 13-Jul-2014 11:08:54 $ 
%% DEVELOPED : 8.3.0.532 (R2014a) 
%% FILENAME  : cmu_raw_to_xyz_centered_new.m 

assert(size(datamap,3) == 3);

oldSize = size(datamap);
datamap = reshape(datamap,oldSize(1)*oldSize(2),3); % first col: depth; second: elevation (altitude); third: azimuth
noResultMask= datamap(:,1)==0;
fprintf('%d total points, %d zero points, %d good points\n', size(datamap,1), sum(noResultMask), sum(~noResultMask));
datamap(:,2) = -(datamap(:,2) - 90); % so that 0 elevation is straight ahead, and 90 elevation is directly overhead.


aziShift = -mean(datamap(~noResultMask,3)); % straight ahead is 0 azimuth. And turning left means positive.

datamap(:,3) = (datamap(:,3)+aziShift); % so that -90 azimuth is just right, and 90 azimuth is just left.

% asser that every point is in front of us... However, this is not true for
% Brown dataset, it seems...
assert(min(datamap(~noResultMask,3)) >= -90);
assert(max(datamap(~noResultMask,3)) <= 90);

assert(min(datamap(~noResultMask,2))>-90);
assert(min(datamap(~noResultMask,2))<90);

% to radian...
datamap(:,2) = datamap(:,2)/180*pi;
datamap(:,3) = datamap(:,3)/180*pi;

% assumption 1: longtitude-azimuth, latitude-elevation

% you can think this as alternation of MATLAB's sph2cart function, with
% directions of coordinates changed. See README.md for more.

% rearrange datamap in the order of azi, elev, and r.
datamap = datamap'; % it's now 3 x N.
datamap = datamap([3 2 1],:);

xyzMatrix = stereo.retina22cart(datamap);
xyzMatrix = xyzMatrix'; % N x 3.

xyzMatrix(noResultMask,:) = NaN;

xyzMatrix = reshape(xyzMatrix,oldSize); % a 3-D matrix again.


end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [cmu_raw_to_xyz_centered_new.m] ======  
