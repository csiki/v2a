% run like:
% myCluster = parcluster('local');
% myCluster.NumWorkers = 7;
% saveProfile(myCluster);
% job = batch('convertAllInFolder2DPar', 'Pool', 6);
% wait(job); delete(job); clear job;

%folderpath = 'e:\img_align_celeba\'
%wildcard = '*.jpg'
%outpath = 'e:\img_align_celeba_corf\'

folderpath = '/media/viktor/0C22201D22200DF0/hand_gestures/own/table3/bw/';
wildcard = '*.png';
outpath = '/media/viktor/0C22201D22200DF0/hand_gestures/own/table3/v1/';

files=dir(fullfile(folderpath, wildcard)); % like path/to/, *.jpg
parfor k=1:length(files)
    imgpath = fullfile(folderpath, files(k).name);
    imgoutpath = fullfile(outpath, strcat('a_', files(k).name));
    if ~(exist(imgoutpath, 'file') == 2)
        respmap = convertImage(imread(imgpath)); % celebA originally is 178x218
        %imwrite(imcomplement(sum(respmap, 3)), imgoutpath);
        imwrite(respmap, imgoutpath);
    % else
    %     display(imgoutpath);
    end
    %if mod(k, 100) == 0
    %    display(k);
    %end
end

% display('done');
