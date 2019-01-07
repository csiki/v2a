function hashkey = getHashKey(vector)

primelist = [2 3 5 7 11 13];
n = numel(vector);

% based on prime factorization
hashkey = prod(primelist(1:n).^vector);
