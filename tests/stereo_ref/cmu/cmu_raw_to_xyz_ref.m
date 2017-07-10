function cmu_raw_to_xyz_ref()
%BROWN_RAW_TO_SPH_AND_XYZ_REF Summary of this function goes here
%   Detailed explanation goes here
matFromLee = load('./04.mat');
datamap = matFromLee.Data(:,:,[1 6 7]);
    
[xyzMatrixArray,noResultMask,aziShiftArray] = cmu_raw_to_xyz_centered_new(datamap);

save('./cmu_raw_to_xyz_ref.mat', 'xyzMatrixArray', ...
    'noResultMask', 'aziShiftArray');

end

