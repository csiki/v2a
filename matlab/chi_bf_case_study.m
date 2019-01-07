ox=[repmat(1,15,1); repmat(2,30,1); repmat(3,15,1); repmat(4,9,1); repmat(5,15,1)]; % number of guesses
a=1; % lower boundary
b=5; % higher boundary

N=length(ox); % sample size

x=unifrnd(a,b,N,1);
nbins = 5; % number of bin
edges = linspace(a,b,nbins+1); % edges of the bins
E = N/nbins*ones(nbins,1); % expected value (equal for uniform dist)

[h,p,stats] = chi2gof(ox,'Expected',E,'Edges',edges)

% not needed
h = histogram(ox,edges);
chi = sum((h.Values - N/nbins).^2 / (N/nbins));
k = nbins-1; % degree of freedom
chi2cdf(chi, k)
