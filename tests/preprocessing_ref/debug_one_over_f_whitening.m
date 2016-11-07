function debug_one_over_f_whitening()
%DEBUG_ONE_OVER_F_WHITENING Summary of this function goes here
%   Detailed explanation goes here

rng(0,'twister');
% original image are of size 128x256
original_images = randn(128,256,5);
new_images_default = whiten_olsh_lee_wrapper(original_images, [], [], []);
% 40 is arbitrary.
new_images_change_f0 = whiten_olsh_lee_wrapper(original_images, 40, [], []);
new_images_no_cutoff = whiten_olsh_lee_wrapper(original_images, inf, [], []);
new_images_change_crop = whiten_olsh_lee_wrapper(original_images, [], [64, 128], []);
new_images_change_crop_pure = whiten_olsh_lee_wrapper(original_images, [], [64, 128], false);


%% save result
save_file_name = 'one_over_f_whitening_ref.hdf5';
save_one_case(save_file_name, '/original_images', original_images);
save_one_case(save_file_name, '/new_images_default', new_images_default);
save_one_case(save_file_name, '/new_images_change_f0', new_images_change_f0);
save_one_case(save_file_name, '/new_images_no_cutoff', new_images_no_cutoff);
save_one_case(save_file_name, '/new_images_change_crop', new_images_change_crop);
save_one_case(save_file_name, '/new_images_change_crop_pure', new_images_change_crop_pure);

end

function new_images = whiten_olsh_lee_wrapper(images, f_0, central_crop, filter_flag)

[N1, N2, N3] = size(images);
if ~isempty(central_crop)
    N1 = central_crop(1);
    N2 = central_crop(2);
end

new_images = zeros(N1, N2, N3);

for iImg = 1:N3
    im_this = images(:,:,iImg);
    new_im_this = whiten_olsh_lee_inner(im_this, f_0, central_crop, filter_flag);
    new_im_this_debug = whiten_olsh_lee_inner(im_this-mean(im_this(:)), f_0, central_crop, filter_flag);
    if filter_flag
        assert(max(abs(new_im_this(:) - new_im_this_debug(:))) < 1e-10);
    end
    new_images(:,:,iImg) = new_im_this;
end

end

function save_one_case(save_file_name, save_group, images)
h5create(save_file_name, save_group, size(images), 'DataType', 'double');
h5write(save_file_name, save_group, images);
end
