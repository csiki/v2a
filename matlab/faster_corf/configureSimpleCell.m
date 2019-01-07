function filterModel = configureSimpleCell(sigma,sigmaRatio,t2)

% create a synthetic stimulus of a vertical edge
stimulus = zeros(200);
stimulus(:,1:100) = 1;

params.eta = 1;
params.radius = ceil(sigma*2.5)*2+1;

% Apply DoG filter
DoG = zeros(200, 200, 2); % NEEDED TO C CODE GENERATION
DoG(:,:,1) = real(getDoG(stimulus, sigma, 0, sigmaRatio, 1, 0));
DoG(:,:,2) = -DoG(:,:,1);

% Half wave rectification
DoG(:,:,1) = DoG(:,:,1) .* (DoG(:,:,1) > 0);
DoG(:,:,2) = DoG(:,:,2) .* (DoG(:,:,2) > 0);
    
% Choose rho list according to given sigma value
if sigma >= 1 && sigma <= 2
    params.rho = [14.38 6.9796 3.0310 1.4135];
elseif sigma > 2 && sigma < 2.4
    params.rho = [19.18 9.369 4.5128 2.1325];
elseif sigma >= 2.4 && sigma <= 3.5
    params.rho = [24.62 12.6488 6.1992 3.0515];
elseif sigma >= 4 && sigma <= 5
    params.rho = [34.43 18.08 9.2467 4.7877 3.3021];
end
% YEP
params.alpha = 0.9;
params.sigma0 = 2;
params.sigmaRatio = sigmaRatio;
params.blurringType = 'Sum';
    
% Create configuration set
fp = [100,100];
simpleCell.inhibition = zeros(4,14);
simpleCell.excitation = zeros(length(params.rho), 14);
onoff_counter = 1;
for r = 1:length(params.rho)
    [onoff, rho, phi] = determineProperties(DoG,params.rho(r),params.eta,fp,t2);
    if ~isempty(onoff)
        simpleCell.excitation(:, round(onoff_counter):round(onoff_counter+length(onoff)-1)) = [onoff; repmat(sigma,1,length(onoff)); rho; phi];
        onoff_counter = onoff_counter + length(onoff);
    end        
end
simpleCell.excitation(4,:) = mod(simpleCell.excitation(4,:),2*pi);

filterModel.params = params;
filterModel.params.radius = round(max(params.rho) + (2 + params.alpha*max(params.rho))/2);
filterModel.simpleCell = simpleCell;

function [onoff, rho, phi] = determineProperties(input,rho,eta,fp,t2)
gb = max(input,[],3);
t = inf;
for i = 1:size(input,3)    
    mxlist = max(max(input(:,:,i)));
    if mxlist < t
        t = mxlist;
    end
end
t = min(mxlist);

x = 1:360*2;

if rho > 0 
    y = gb(sub2ind([size(input,1),size(input,2)],round(fp(1) + rho.*cos(x.*pi/180)),round(fp(2)+(rho.*sin(x.*pi/180)))));
    y = circshift(y',270)';
    if length(unique(y)) == 1
        onoff = [];
        rho = [];
        phi = [];                
        return;
    end

    y(y < 0.01) = 0;   
    y = round(y*1000)/1000;
    BW = bwlabel(imregionalmax(y));
    npeaks = max(BW(:));
    peaks = zeros(1,npeaks);
    for i = 1:npeaks
        peaks(i) = mean(find(BW == i));
    end

    f = peaks >= 180 & peaks < 540;
    phivalues = peaks(f);
    % this shit never runs
%     if phivalues(1)+360 - phivalues(end) < eta
%         if P(locs(uidx(1))) < P(locs(uidx(end)))
%             phivalues(1) = [];
%         else
%             phivalues(end) = [];
%         end
%     end
    
    [x, y] = pol2cart((phivalues-270)*pi/180,rho);    
    phi = phivalues*pi/180;
    onoff = zeros(1,numel(phi));
    for i = 1:numel(phi)
        [ignore idx] = max(input(round(fp(2)+x(i)),round(fp(1)+y(i)),:));
        onoff(i) = idx-1;
    end
else
    centreResponses = reshape(input(round(fp(2)),round(fp(1)),:),1,2);    
    mx = max(centreResponses,[],1);
    if max(mx) >= t
        onoff = find(mx > t2*max(mx)) - 1;
        phi = zeros(1,length(onoff));
    else
        onoff = [];
        phi = [];
    end
end
rho = repmat(rho,1,length(phi));