function imw = whiten_olsh_lee_inner(im, f_0, central_crop, filter_flag)

% this function implements 1/f whitening, adpated from code of Olshausen
% and code from Honglak Lee

% original code from Olshausen
% from `make-your-own-images` of `sparsenet.tar.gz` of
% http://redwood.berkeley.edu/bruno/sparsenet/
%   N=image_size;
%   M=num_images;
%
%   [fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
%   rho=sqrt(fx.*fx+fy.*fy);
%   f_0=0.4*N;
%   filt=rho.*exp(-(rho/f_0).^4);
%
%   for i=1:M
%     image=get_image;  % you will need to provide get_image
%     If=fft2(image);
%     imagew=real(ifft2(If.*fftshift(filt)));
%     IMAGES(:,i)=reshape(imagew,N^2,1);
%   end
%
%   IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));
%
%   save MY_IMAGES IMAGES
%
% original code from Honglak Lee
% from `crbm_v1/tirbm_whiten_olshausen2_invsq_contrastnorm.m` of
% https://github.com/honglaklee/convDBN/blob/master
% commit SHA1 11bf1c9891e0bfe936186f26c88cb8f9d8dac8c5
%
%
%
% im = IMAGESr(:,:,3);
%
% N1 = size(im,1);
% N2 = size(im,2);
%
% [fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
% rho=sqrt(fx.*fx+fy.*fy)';
% f_0=0.4*mean([N1,N2]);
% filt=rho.*exp(-(rho/f_0).^2);
%
% imagew=real(ifft2(fft2(im).*fftshift(filt)));

im = double(im);

N1 = size(im, 1);
N2 = size(im, 2);

% make sure they are even
assert(rem(N1,2)==0);
assert(rem(N2,2)==0);

[fx, fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
rho=sqrt(fx.*fx+fy.*fy)'; % this transpose is important...
if nargin < 2 || isempty(f_0)
    f_0=0.4*mean([N1,N2]);
end

filt=rho.*exp(-(rho/f_0).^4);
If=fft2(im);

if nargin < 4 || isempty(filter_flag)
    filter_flag = true;
end
if filter_flag
    fftFilteredOld = If.*fftshift(filt);
else
    fftFilteredOld = If;
end
fftFilteredOld = fftshift(fftFilteredOld);
if nargin < 3 || isempty(central_crop)
    central_crop = [N1, N2];
end
fftFilteredOld = fftFilteredOld(...
    (N1/2-central_crop(1)/2+1):(N1/2+central_crop(1)/2),...
    (N2/2-central_crop(2)/2+1):(N2/2+central_crop(2)/2));


fftFilteredOld = ifftshift(fftFilteredOld);

imw=real(ifft2(fftFilteredOld));
% take real, although the complex part should be very very small.
% since fft/ifft is linear operation, mean of new image is also zero.



end