%function corfresponse = convertImage(imgpath)
function corfresponse = convertImage(img)

    corfresponse = CORFContourDetection(img, 2.2, 4, 1.8, 0.007);
    %corfresponse = max(corfresponse, [], 3);
    % for i = 1:downscale
    %     respmap = downscaleResp(respmap);
    % end
    
end