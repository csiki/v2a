function [output sigma] = getDoG(img,sigma, onoff, sigmaRatio, config, width, threshold)
% create Difference of Gaussian Kernel
sz = size(img) + width + width;    
% sz = ceil(sigma*3) * 2 + 1;

g1 = fspecial('gaussian',sz,sigma);
g2 = fspecial('gaussian',sz,sigma*sigmaRatio);

if onoff == 1
    G = g2 - g1;  
else
    G = g1 - g2;
end

%G(G<0) = G(G<0) .* abs(sum(G(G>0))/sum(G(G<0)));

resultimg = padarray(img,[width width],'both','symmetric');

% compute DoG
output = fftshift(ifft2(fft2(G,sz(1),sz(2)) .* fft2(resultimg)));
%output = conv2(single(resultimg),single(G),'same');

% trim and threshold
% [imH imW] = size(img);
% from = ([imH imW]./2)+1;
% output = output(from(1)-width:from(1)+imH-1+width,from(2)-width:from(2)+imW-1+width);
if nargin == 7    
    output(output < threshold*max(output(:))) = 0;
end
