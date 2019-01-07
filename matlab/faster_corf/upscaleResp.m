function upscaled = upscaleResp(respmap)
    
    % TODO upscale the image by 2 and spread the respmap uniformly in it
    % TODO use 3x3 matrix to imprint the orientations additively
    % hardcoded for 45 degrees
    norientation = size(respmap, 3);
    orientMatrices = zeros(3, 3, norientation);
    orientMatrices(:,:,1) = [0,1,0;0,1,0;0,1,0];
    orientMatrices(:,:,2) = [1,0,0;0,1,0;0,0,1];
    orientMatrices(:,:,3) = [0,0,0;1,1,1;0,0,0];
    orientMatrices(:,:,4) = [0,0,1;0,1,0;1,0,0];
    
    upscaled = zeros(size(respmap, 1) * 2, size(respmap, 2) * 2, size(respmap, 3));
    upscaled(1:2:end, 1:2:end, :) = respmap; % spread
    
    % "deconvolve" = just convolve, then have a threshold of >0
    for i = 1:norientation
        upscaled(:,:,i) = conv2(upscaled(:,:,i), orientMatrices(:,:,i), 'same');
    end
    
    [argval, argmax] = max(upscaled, [], 3);
    
    for i = 1:norientation
        upscaled(:,:,i) = (argmax == i) .* (argval > 0);
    end
    
end