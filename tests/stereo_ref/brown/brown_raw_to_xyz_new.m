function [xyzMatrix,noResultMask,aziShift,ddp,datamap] = brown_raw_to_xyz_new(datamap)
% BROWN_RAW_TO_XYZ_NEW ... 
%  
%   rewritten version to convert raw Brown dataset to XYZ. 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 06-Apr-2014 17:12:07 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : brown_raw_to_xyz_new.m 

assert(size(datamap,3) == 3);

% I don't exacly what kind of coordinate system is used, e.g. is
% equi-altitude points lie on great circles, or circles with different
% radius? (the same problem for azimuth).

% 
% About the variables in the .mat files:
% 
%   Every range image is stored as a structure with the following organization:
% 
%   r.range_m     distance (polar coordinates!) in meters to object
%                 A zero value means that the data point is missing; 
%                 This will, for example, happen when an object is out
%                 of range. 
% 
%   r.intensity_unk   reflectance image for laser range-finder
%                     (again, zero values where data is missing)
% 
%   bearing_rad       angles for vertical direction,
%                     NOTE: Should be multiplied with 2 for radians!
%                     The LRF starts scanning at about 0.89 rad or 51 deg, 
%                     and ends at about 2.28 rad or 131 deg 
%                     (0 is up, pi/2 is straight forward, and pi is down).
% 
% 
%  inclination_rad   angles for horizontal direction (in radians)
%  looks that this variable is centered at 2.7 (a little smaller than pi),
%  and actually it spans over 180 degrees.
%  looks that this thing goes bigger from left to right.

% assume great circle for azimuth, and parallel circle for altitude.

% I think this is correct; because for images with a rectangle in it, I
% really get a rectangle back, without any bening & distortion on edges.
% if my assumption is wrong, I won't get a upright rectangle back.

oldSize = size(datamap);
% display(oldSize);

% ddp is a two element vector, first being deg per pixel for azimuth,
% second for elevation.

% in this one, datamap(:,:,3) is azimuth.

ddp = zeros(2,1);

azimuthDiff = diff(datamap(:,:,3),1,2);
azimuthDiff = azimuthDiff(:);
assert(all(~isnan(azimuthDiff)));
ddp(1) = mean(azimuthDiff)*180/pi;

elevationDiff = diff(2*datamap(:,:,2),1,1); % times 2 for correction, as noted in original documentation.
elevationDiff = elevationDiff(:);
assert(all(~isnan(elevationDiff)));
ddp(2) = mean(elevationDiff)*180/pi;

datamap = reshape(datamap,oldSize(1)*oldSize(2),3); % first col: depth; second: elevation (altitude); third: azimuth

noResultMask= datamap(:,1)==0;

assert(all(datamap(:,1)>=0));

fprintf('%d total points, %d zero points, %d good points\n', size(datamap,1), sum(noResultMask), sum(~noResultMask));

datamap(:,2) = -(2*datamap(:,2) - pi/2); % so that 0 elevation is straight ahead, and 90 elevation is directly overhead.

datamap(:,3) = (- datamap(:,3)); % so that 0 azimuth is just right, and 180 azimuth is just left.

aziShift = - mean(datamap(:,3));
% aziShift = - mean(datamap(:,3)) + pi/2;

datamap(:,3) = datamap(:,3) + aziShift;
% shifting the azimuth map so that things with the mean azi will have zero azimuth, 
% which correspond to being in the front for retina2 system.

% asser that every point is in front of us... However, this is not true for
% Brown dataset, it seems...
% assert(min(datamap(~noResultMask,3)) >= 0); 
% assert(max(datamap(~noResultMask,3)) <= 2*pi);

assert(min(datamap(~noResultMask,3)) >= -pi); 
assert(max(datamap(~noResultMask,3)) <= pi);

% this should work
assert(min(datamap(~noResultMask,2))>=-pi/2);
assert(min(datamap(~noResultMask,2))<=pi/2);

% % to radian...
% datamap(:,2) = datamap(:,2)/180*pi;
% datamap(:,3) = datamap(:,3)/180*pi;

% assumption 1: longtitude-azimuth, latitude-elevation

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
% ===== EOF ====== [brown_raw_to_xyz_new.m] ======  
