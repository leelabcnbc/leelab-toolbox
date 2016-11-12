% how to use: unzip rsatoolbox, and run this script under tests/rsa_ref.
rng(0,'twister');
addpath(genpath(fullfile(pwd, 'rsatoolbox')));
% 10 test cases.
nCase = 10;
nRDM = 5;
nSample = 100;
% create 5 random feature matrices, of size 100 x 200.
feature_matrix_all = randn(nSample, 200, nRDM, nCase);
rdm_stack_all = zeros(nSample,nSample,nRDM, nCase);
% construct RDM stack.
result_array = zeros(nCase,2);
for iCase = 1:nCase
    feature_matrix_all_this_case = feature_matrix_all(:,:,:,iCase);
    rdm_stack = zeros(nSample,nSample,nRDM);
    for iFeatureMatrix = 1:nRDM
        rdmThis = squareform(pdist(feature_matrix_all_this_case(:,:,iFeatureMatrix), ...
            'correlation'));
        rdm_stack(:,:,iFeatureMatrix) = rdmThis;
    end
    rdm_stack_all(:,:,:,iCase) = rdm_stack;
    % compute upper and lower bounds.
    [ceiling_upperBound, ceiling_lowerBound] = ...
        ceilingAvgRDMcorr(rdm_stack,'Spearman',false);
    result_array(iCase,1) = ceiling_upperBound;
    result_array(iCase,2) = ceiling_lowerBound;
end

% save the result.
% feature_matrix_all can be used to do check computation of my RDM.
h5create('rsa_ref.hdf5', '/rsa_bounds/feature_matrix_all', size(feature_matrix_all));
h5write('rsa_ref.hdf5', '/rsa_bounds/feature_matrix_all', feature_matrix_all);
h5create('rsa_ref.hdf5', '/rsa_bounds/rdm_stack_all', size(rdm_stack_all));
h5write('rsa_ref.hdf5', '/rsa_bounds/rdm_stack_all', rdm_stack_all);
h5create('rsa_ref.hdf5', '/rsa_bounds/result_array', size(result_array));
h5write('rsa_ref.hdf5', '/rsa_bounds/result_array', result_array);