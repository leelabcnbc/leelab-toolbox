% test that my program for computing disparity is correct, compared to
% Xiong's

% passed 20140713T231202
clear all;
close all;
load('V1_11_xyz.mat'); % XYZ matrix for the scene used in Xiong's demo.
fixationIndex = [256, 1238];
rowIndex = fixationIndex(1)-100:fixationIndex(1)+100;
colIndex = fixationIndex(2)-100:fixationIndex(2)+100;

fixationThis = V1_11_xyz(fixationIndex(1),fixationIndex(2),:);
fixationThis = fixationThis(:);
% this 3.9280 is to handle medium filtering in Xiong's code.
fixationThis = fixationThis./norm(fixationThis)*3.9280;
patchXYZ = V1_11_xyz(rowIndex,colIndex,:);

patchXYZ_reshaped = reshape(patchXYZ,[201*201,3])';


retinaImage = stereo.cart2retina2(patchXYZ_reshaped);
retinaImage = reshape(retinaImage(1,:),201,201);

aziDiff = diff(retinaImage,[],2);
ddp = nanmean(aziDiff(:))*180/pi;

patchDisp_reshaped = stereo.cart2disparity(patchXYZ_reshaped,fixationThis);
patchDisp_reshaped = reshape(patchDisp_reshaped,201,201);

rangeMap = sqrt(sum(patchXYZ.^2,3));
% compared to Xiong's disparity computation, mine use radians, and the
% signs are reversed. 
% ZYM: this reversed is not due to Yang Liu's equation, but Xiong's wrong
% implementation (fixZ are all positve in Xiong's case, and all negative in
% Liu's case).
patchDisp_reshaped = -patchDisp_reshaped./pi*180;

V1_11_range = sqrt(sum(V1_11_xyz.^2,3));

V1_11_range = V1_11_range(rowIndex,colIndex);

% aziDiff = diff(retinaImage,[],2);
% ddp = nanmean(aziDiff(:))*180/pi;
% just write another function to wrap Xiong's function.
patchDisp_reshaped_xiong = range2disparityXiong(V1_11_range,5.56);

imagesc(abs(patchDisp_reshaped_xiong-patchDisp_reshaped)); colorbar;
save('test_xiong_computation.mat', 'patchDisp_reshaped', 'patchDisp_reshaped_xiong');