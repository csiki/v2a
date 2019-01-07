% proper table dataset trained model stats
% score: 1 nohit, 2 touch w/ at least 2 fingers, 3 grip covers the center
s = [2,3,1,3,1,3,1,2,2,1,2,2,1,1,3,2,1,3,3,2,3,2,3,2,3,2,2,3,3,1,3,3,3,3,2,1,1,1,3,3];
% 1: beer can, 2: gear
o = [1,2,1,1,2,2,1,1,2,2,2,1,2,1,2,2,1,2,1,1,2,2,2,1,1,2,2,1,2,2,2,1,2,1,2,1,2,1,1,2];
% 1: right object, 2: wrong object
r = [1,1,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,1,0,0,1,1,1];
% first 8 missing, and 2 other
t = [0,0,0,0,0,0,0,0,11,14,19,8,11,13,9,0,14,16,5,17,18,10,14,8,13,0,15,10,13,21,14,12,25,10,23,16,35,22,24,19];

% wrong hand posture trained model stats
sw = [3,1,1,1,1,1,1,1,1,1,1,2,1,3,1,1,2,1,3,1,3,1,1,1,1,1,1,1,1,1,1,1,2,2,2,1,2,3,1,1];
ow = [1,2,1,1,2,2,2,1,1,2,1,2,1,2,1,2,2,1,2,2,1,1,2,1,2,1,2,1,1,2,2,2,1,1,2,2,1,2,1,1];
rw = [1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0];

% statistics
valid_t = t(t>0);
mean_time = mean(valid_t);
std_time = std(valid_t);
%hist(valid_t, 10);

% independent t-test of scores
%right = [mean(s), std(s)]
%wrong = [mean(sw), std(sw)]
%[h,p,ci,stats] = ttest2(s,sw)

% chi-square goodness of fit test of object identification accuracy
chi2test([r;rw], 2);

function chi2test(data, num_int)
	[m,n] = size(data)
	
	E = m/num_int; %E=10
	interval=zeros(num_int, 1);
	for i=1:m
		k = data(i,1);
		interval(fi(num_int,k),1) = interval(fi(num_int,k),1) +1;
	end
	x = 0;
	for j= 1:num_int
		x=x+ (interval(j,1)-E)^2/E;
	end
	x
	chi2inv(0.95,num_int-1)
	
function b = fi(i,n)
    b=1;
for j= 0:i
	if n > (1/i)*(i-j)
		b = i-j+1;
		break
	end
end
end
end