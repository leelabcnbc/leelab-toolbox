function brown_raw_to_sph_and_xyz_ref()
%BROWN_RAW_TO_SPH_AND_XYZ_REF Summary of this function goes here
%   Detailed explanation goes here
matFromLee = load('./V3_4.mat');
datamap = [matFromLee.r.range_m,matFromLee.r.bearing_rad,matFromLee.r.inclination_rad];
    
datamap = reshape(datamap,size(matFromLee.r.range_m,1),size(matFromLee.r.range_m,2),3);
    
[xyzMatrixArray,noResultMask,aziShiftArray,ddpArray,datamap] = brown_raw_to_xyz_new(datamap);

save('./brown_raw_to_sph_and_xyz_ref.mat', 'xyzMatrixArray', ...
    'noResultMask', 'aziShiftArray', 'ddpArray', 'datamap');

end

