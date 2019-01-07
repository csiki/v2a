function downscaled = downscaleResp(respmap)
    
    % orientMatrices as 3x3xorientations to conv with
    % hardcoded for 45 degrees
    norientation = size(respmap, 3);
    orientMatrices = zeros(3, 3, norientation);
    orientMatrices(:,:,1) = [0,1,0;0,1,0;0,1,0];
    orientMatrices(:,:,2) = [1,0,0;0,1,0;0,0,1];
    orientMatrices(:,:,3) = [0,0,0;1,1,1;0,0,0];
    orientMatrices(:,:,4) = [0,0,1;0,1,0;1,0,0];
    
    %cf = ones(3);
    downscaled = zeros(size(respmap));
    for i = 1:norientation
        downscaled(:,:,i) = conv2(respmap(:,:,i), orientMatrices(:,:,i), 'same');
    end
    
    downscaled = downscaled(1:2:end, 1:2:end, :);
    [argval, argmax] = max(downscaled, [], 3);
    
    for i = 1:norientation
        downscaled(:,:,i) = (argmax == i) .* (argval > 0);
    end
    
end