% CORFContourDetection Compute contour map
%
% Input:
%     img   - a coloured or grayscale image
%     sigma - The standard deviation of the DoG functions used
%     beta  - The increase in the distance between the sets of center-on
%             and ceter-off DoG receptive fields.
%     inhibitionFactor - The factor by which the response exhibited in the
%                        inhibitory receptive field suppresses the response exhibited in the
%                        excitatory receptive field
%     highthresh - The high threshold of the non-maximum suppression used
%                  for binarization
% Output:
%     binarymap - The binary map containing the contours
%     corfresponse - The response map of the rotation-invariant CORF operator
%
% Usage example: 
%       [binmap, corfresponse] = CORFContourDetection(imread('./img/134052.jpg'),2.2,4,1.8,0.007);
%       figure;imagesc(imcomplement(binmap));axis image;axis off;colormap(gray);
%       figure;imagesc(corfresponse);axis image;axis off;colormap(gray);
%
%       [binmap, corfresponse] = CORFContourDetection(imread('./img/rino.pgm'),2.2,4,1.8,0.005);
%       figure;imagesc(imcomplement(binmap));axis image;axis off;colormap(gray);
%       figure;imagesc(corfresponse);axis image;axis off;colormap(gray);
%
% You are invited to use this Matlab script and cite the following articles:
%       Azzopardi G, Rodr�guez-S�nchez A, Piater J, Petkov N (2014) A Push-Pull CORF Model of a Simple Cell 
%           with Antiphase Inhibition Improves SNR and Contour Detection. PLoS ONE 9(7): e98424. 
%           doi:10.1371/journal.pone.0098424
%
%       Azzopardi G, Petkov N (2012) A CORF Computational Model of a Simple Cell that relies on LGN Input 
%           Outperforms the Gabor Function Model. Biological Cybernetics 1?13. doi: 10.1007/s00422-012-0486-6

function corfresponse = CORFContourDetection(img,sigma, beta, inhibitionFactor, highthresh)

%%%%%%%%%%%%%%%% BEGIN CONFIGURATION %%%%%%%%%%%%%%%%%%
% Add Utilities folder to path

if ndims(img) == 3
    img = rgb2gray(img);
end

img = rescaleImage(double(img),0,1);

% Create CORF model simple cell
CORF = configureSimpleCell(sigma,0.5,0.5);

% create the inhibitory cell
%CORF.simpleCell.inhibition = modifyModel(CORF.simpleCell.excitation,'invertpolarity',1,'overlappixels',beta);

inh_model = CORF.simpleCell.excitation;
inh_model(1,:) = 1 - inh_model(1,:);

rho = inh_model(3,:);
phi = inh_model(4,:);
[x, y] = pol2cart(phi,rho);
negx = x < 0;        
x(negx) = x(negx) - beta;
x(~negx) = x(~negx) + beta;
[phi, inh_model(3,:)] = cart2pol(x,y);
inh_model(4,:) = mod(phi+2*pi,2*pi);

CORF.simpleCell.inhibition = inh_model;

%model(1,:) = 1-model(1,:);
%%%%%%%%%%%%%%%% END CONFIGURATION %%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% BEGIN APPLICATION %%%%%%%%%%%%%%%%%%%
orienslist = 0:45:350;
%CORF.params.blurringType = 'Sum';

% compute CORF response image for each orientation given in orienslist
output = getSimpleCellResponse(img,CORF,orienslist,inhibitionFactor);

% combine the output of all orientations
%[corfresponse, oriensMatrix, oriensMatrixIndex] = calc_viewimage(output,1:numel(orienslist), orienslist*pi/180);
% COMMENTED OUT NOT NEEDED, THE output IS THE ONLY ONE USED

%%%%%%%%%%%%%%%%% END APPLICATION %%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% BEGIN BINARIZATION %%%%%%%%%%%%%%%%%%
% compute thinning
% [thinning, thinnedOrientation] = calc_thinning(corfresponse, oriensMatrix, oriensMatrixIndex, 1);
% 
% % Choose high threshold of hysteresis thresholding
% if nargin == 4
%     bins = 64; p = 0.1; %Keep the strongest 10% of the pixels in the resulting thinned image
%     f = find(thinning > 0);
%     counts = imhist(thinning(f),bins);
%     highthresh = find(cumsum(counts) > (1-p)*length(f),1,'first') / bins;
% end
% 
% binarymap = calc_hysteresis(thinning, 1, 0.5*highthresh, highthresh);
% TODO commented out until here

% compute the binary response at each orientation after thinning
% TODO mod pi the number of orientations in the orientation matrix
% binaryResponseMap = zeros(size(output));
% binarymap = zeros(size(output));

% LOOK OUT: corfresponse hack
% nrealorientation = length(orienslist) / 2;
corfresponse = (1 / max(output(:))) * output;
% for i = 1:(nrealorientation)
%     corfresponse(:, :, i) = max(corfresponse(:, :, i), corfresponse(:, :, i+nrealorientation));
% end
% corfresponse = corfresponse(:,:,1:nrealorientation);
corfresponse = max(corfresponse, [], 3);

% TODO FIXME COMMENTED OUT !!!
% separate different orientations to different layers
% filter each layer by the already computed
% for i = 1:numel(orienslist)
%     binaryResponseMap(:,:,i) = (thinnedOrientation == i) .* binarymap;
% end
% 
% % mod pi the orientation, and add layers of same orientation
% for i = 1:(nrealorientation)
%     binaryResponseMap(:,:,i) = (binaryResponseMap(:,:,i) + binaryResponseMap(:,:,i+nrealorientation)) > 0;
% end
% binaryResponseMap = binaryResponseMap(:,:,1:nrealorientation);
% TODO UNTIL HERE




% summedBinRespMap = sum(binaryResponseMap, 3);
% show binarized image
% figure;
% subplot(1,3,1);imagesc(img);axis off;axis image;colormap(gray);
% subplot(1,3,2);imagesc(imcomplement(binarymap));axis off;axis image;colormap(gray);
% subplot(1,3,3);imagesc(imcomplement(summedBinRespMap));axis off;axis image;colormap(gray);
%%%%%%%%%%%%%%%%% END BINARIZATION %%%%%%%%%%%%%%%%%%%%

% figure;
% for i=1:size(binaryResponseMap, 3)
%     subplot(2,2,i);imagesc(imcomplement(binaryResponseMap(:,:,i)));axis off;axis image;colormap(gray);
% end