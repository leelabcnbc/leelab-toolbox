function dsp = range2disparityXiong(rim,ppd,bdist)
% RANGE2DISPARITYXIONG ...
%
%   ...
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 15-Jul-2014 10:26:36 $
%% DEVELOPED : 8.3.0.532 (R2014a)
%% FILENAME  : range2disparityXiong.m
if nargin < 3 || isempty(bdist)
    bdist = 0.0325; % half of IODistance
end

rfix = round( (1+size(rim,1)) /2);
cfix = round( (1+size(rim,2)) /2);

[~, colInd] = ind2sub(size(rim),1:numel((rim))); % this colInd is more important.


colInd = reshape(colInd,size(rim));

% get the disparity of the fixiation point
reg = rim(rfix-3:rfix+3,cfix-3:cfix+3);
fixZ = dispfilter(reshape(reg,[numel(reg),1]),0);

alpha = 2 * atan(-bdist/fixZ) * 180/pi;

xp = sin((colInd-cfix)./ppd * pi/180).*rim;
zp = cos((colInd-cfix)/ppd * pi/180).*rim;
phi = (atan((-xp-bdist)./zp) - atan((-xp+bdist)./zp)) * 180/pi;

dsp = alpha - phi;


end


function [dsps] = dispfilter(patches,bAver)
dsps = zeros(1,size(patches,2));
for i = 1:size(patches,2)
    patch = patches(:,i);
    patch = patch(~isnan(patch));
    
    if isempty(patch)
        continue;
    end
    
    if bAver
        dsp = mean(patch);
    else
        patch = sort(patch);
        dsp = patch(round(length(patch)/2));
    end
    
    dsps(i) = dsp;
end

end






% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [range2disparityXiong.m] ======
