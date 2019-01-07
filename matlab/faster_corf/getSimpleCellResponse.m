function [response, responseParams] = getSimpleCellResponse(img,model,orientlist,inhibitionFactor,responseParams)

% Compute all blurred responses and store the results in a hash table for
% fast retrieval. This can be implemented in a parallel mode.
if nargin == 4
    [SimpleCellHashTable, weightsigma] = computeBlurredResponse(img,model);
    responseParams.SimpleCellHashTable = SimpleCellHashTable;
    responseParams.weightsigma = weightsigma;
else
    SimpleCellHashTable = responseParams.SimpleCellHashTable;
    weightsigma = responseParams.weightsigma;
end    

response = zeros(size(img,1),size(img,2),numel(orientlist));
for i = 1:numel(orientlist)    
    rotmodel.simpleCell.excitation = modifyModel(model.simpleCell.excitation,'thetaoffset',orientlist(i)*pi/180);
    if isfield(model.simpleCell,'inhibition')
        rotmodel.simpleCell.inhibition = modifyModel(model.simpleCell.inhibition,'thetaoffset',orientlist(i)*pi/180);    
    end

    % Compute the excitatory response of the simple cell
    excitationresponse = getResponse(SimpleCellHashTable,rotmodel.simpleCell.excitation,weightsigma,model.params);

    if isfield(model.simpleCell,'inhibition')
        % Compute the antiphase inhibitory response of the simple cell
        inhibitionresponse = getResponse(SimpleCellHashTable,rotmodel.simpleCell.inhibition,weightsigma,model.params);

        % Compute the net response
        rotresponse = excitationresponse - inhibitionFactor*inhibitionresponse;
        rotresponse = rotresponse .* (rotresponse > 0);
    else
        % If no inhibition, then we only take the excitatory response
        rotresponse = excitationresponse;
    end
    response(:,:,i) = rotresponse;
end

function response = getResponse(SimpleCellHashTable,simpleCell,weightsigma,params)
ntuples = size(simpleCell,2);
w = zeros(1,ntuples);
response = 1;

for i = 1:ntuples
    delta = simpleCell(1,i);
    sigma = simpleCell(2,i);
    rho   = round(simpleCell(3,i)*1000)/1000;
    phi   = simpleCell(4,i);
    
    [col, row] = pol2cart(phi,rho);
    blurredResponse = SimpleCellHashTable(getHashKey([delta sigma rho]));
    shiftedResponse = imshift(blurredResponse,fix(row),-fix(col));                                                        
    w(i) = exp(-rho^2/(2*weightsigma*weightsigma));        
    response = response .* shiftedResponse;    
end
response = response.^(1/sum(w));
%response = response .^ (1/ntuples);
response = response(params.radius+1:end-params.radius,params.radius+1:end-params.radius);

function [SimpleCellHashTable, weightsigma] = computeBlurredResponse(img,model)
sigmaList = model.simpleCell.excitation(2,:);
paramsList = model.simpleCell.excitation([1 2 3],:)';
if isfield(model.simpleCell,'inhibition')
    sigmaList = [sigmaList model.simpleCell.inhibition(2,:)];    
    paramsList = [paramsList; model.simpleCell.inhibition([1 2 3],:)'];
end

sigmaList = unique(sigmaList);
paramsList = unique(round(paramsList*1000)/1000,'rows');

nsigmaList = numel(sigmaList);
nparamsList = size(paramsList,1);

LGNKeyList = cell(1,nsigmaList*2);
LGNValueList = cell(1,nsigmaList*2);
idx = 1;
for s = 1:nsigmaList
    delta = 0;
    LGNValueList{idx} = getDoG(img,sigmaList(s),0,model.params.sigmaRatio,delta,model.params.radius);        
    LGNKeyList{idx} = getHashKey([delta sigmaList(s)]);
    
    delta = 1;
    LGNValueList{idx+1} = -LGNValueList{idx};
    LGNKeyList{idx+1} = getHashKey([delta sigmaList(s)]);
    
    LGNValueList{idx} = LGNValueList{idx} .* (LGNValueList{idx} > 0);
    LGNValueList{idx+1} = LGNValueList{idx+1} .* (LGNValueList{idx+1} > 0);
    idx = idx + 2;
end
LGNHashTable = containers.Map(LGNKeyList,LGNValueList);

SimpleCellKeyList = cell(1,nparamsList);
SimpleCellValueList = cell(1,nparamsList);

weightsigma = max(paramsList(:,3)) / 3;

for p = 1:nparamsList
    delta = paramsList(p,1);    
    sigma = paramsList(p,2);
    rho   = paramsList(p,3);
    
    SimpleCellKeyList{p} = getHashKey([delta sigma rho]);
    
    LGNHashKey = getHashKey([delta sigma]);
    LGN = LGNHashTable(LGNHashKey);
    
    r = round((model.params.sigma0 + model.params.alpha*rho)/2);
    if r > 0    
        if strcmp(model.params.blurringType,'Sum')
            smoothfilter = fspecial('gaussian',[2*r+1,2*r+1],r/3);
            SimpleCellValueList{p} = conv2(LGN,smoothfilter,'same');            
        elseif strcmp(model.params.blurringType,'Max')
            SimpleCellValueList{p} = maxgaussianfilter(LGN,r/3,0,0,[1 1],size(LGN));        
        end
    else
       SimpleCellValueList{p} = LGN; 
    end 
    SimpleCellValueList{p} = SimpleCellValueList{p} .^ exp(-rho^2/(2*weightsigma*weightsigma));
    %SimpleCellValueList{p} = SimpleCellValueList{p} .^ exp(-rho^2/(2*weightsigma*weightsigma));
end
SimpleCellHashTable = containers.Map(SimpleCellKeyList,SimpleCellValueList);