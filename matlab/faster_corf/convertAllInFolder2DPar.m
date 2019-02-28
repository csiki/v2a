% first change folderpath and outpath below, then run script in parallel in Matlab:
% myCluster = parcluster('local');
% myCluster.NumWorkers = 7;  % number of threads
% saveProfile(myCluster);
% job = batch('convertAllInFolder2DPar', 'Pool', 6);  % number of threads-1
% wait(job); delete(job); clear job;


folderpath = '/path/to/scaled/black_and_white/images/';
wildcard = '*.png';
outpath = '/path/to/output/contour/images/';

files=dir(fullfile(folderpath, wildcard));
parfor k=1:length(files)
    imgpath = fullfile(folderpath, files(k).name);
    imgoutpath = fullfile(outpath, strcat('a_', files(k).name));
    if ~(exist(imgoutpath, 'file') == 2)
        respmap = convertImage(imread(imgpath)); % celebA originally is 178x218
        %imwrite(imcomplement(sum(respmap, 3)), imgoutpath);
        imwrite(respmap, imgoutpath);
    end
end
