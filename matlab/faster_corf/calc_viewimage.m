function [result, oriensMatrix, oriensMatrixIndex] = calc_viewimage(matrices, dispcomb, theta)
% VERSION 14/05/04
% CREATED BY: M.B. Wieling and N. Petkov, Groningen University,
%             Department of Computer Science, Intelligent Systems
%
% CALC_VIEWIMAGE: calculates the maximum-superposition of all the matrices stored
% in MATRICES (according to the L-infinity norm). It uses only the matrices for
% which the index is entered in DISPCOMB, e.g. if DISPCOMB contains the values
% 1,2,4: only the first, second and fourth matrix contained in MATRICES are used
% for the superposition. This method also calculates the orientationmatrix (ORIENSMATRIX) 
% which stores the maximum orientation response of each point in the resulting matrix
% (RESULT) - for use in CALC_THINNING. A progressbar of the calculations is also shown. 
%   CALC_VIEWIMAGE(MATRICES, DISPCOMB, THETA) 
%   calculates the single viewing image according to the following parameters
%     MATRICES - the matrices which hold all the convolutions for each orientation
%     DISPCOMB  - the indexes of each matrix (in MATRICES) which should be used for the
%                 superposition  
%     THETA - a list of all the orientations - to create ORIENSMATRIX

% display a progressbar
%h = waitbar(0,'Merging images to single image, please wait... (Step 5/7)');

% initialize values
oriensMatrix = 0;
oriensMatrixIndex = 0;
tmpMaxConv = -Inf;
result = -Inf;
cnt1 = 1;

if (size(dispcomb,2) == 1)
  result = matrices(:,:,dispcomb(1));
else

  % calculate the superposition (L-infinity norm)
  while (cnt1 <= size(dispcomb,2))
    % calculate the maximum orientation-response in each point (based on the absolute values)
    oriensMatrixtmp1 = (abs(matrices(:,:,dispcomb(cnt1))) > tmpMaxConv) .* theta(dispcomb(cnt1));
    oriensMatrixtmp2 = (abs(matrices(:,:,dispcomb(cnt1))) <= tmpMaxConv) .* oriensMatrix;
    oriensMatrix = oriensMatrixtmp1 + oriensMatrixtmp2;
    
    oriensMatrixtmp1Index = (abs(matrices(:,:,dispcomb(cnt1))) > tmpMaxConv) .* dispcomb(cnt1);
    oriensMatrixtmp2Index = (abs(matrices(:,:,dispcomb(cnt1))) <= tmpMaxConv) .* oriensMatrixIndex;
    oriensMatrixIndex = oriensMatrixtmp1Index + oriensMatrixtmp2Index;
    
    tmpMaxConv = max(abs(matrices(:,:,dispcomb(cnt1))), tmpMaxConv);
   
    % calculate the superposition = max of absolute value at each orientation
    result = max(result,abs(matrices(:,:,dispcomb(cnt1))));
    cnt1 = cnt1 + 1;
    %waitbar(cnt1/size(matrices,3)); % update the progressbar
  end
end
%close(h);
 