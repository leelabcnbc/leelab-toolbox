function test_cart2disparity_new_ref()

rng(0,'twister');

numberOfCases = 500;
numberOfPoints = 1000;
IODistance = 0.065;
ImageArray = randn(3,numberOfPoints,numberOfCases);
FArray = rand(3,numberOfCases)*20-10;

results_all = cell(numberOfCases,1);

for iCase = 1:numberOfCases
    thisF = FArray(:,iCase);
    disparityMap1 = stereo.cart2disparity_new(ImageArray(:,:,iCase),thisF,false,IODistance);
    results_all{iCase} = disparityMap1;
end

save('test_cart2disparity_new_ref.mat', 'results_all', 'FArray', 'ImageArray');

end