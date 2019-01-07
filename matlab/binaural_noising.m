% calculations to aid a simple noising binaural model
% using only three datapoints of recorded localization error a curve is
% fitted. localization error at a azimuth parametrize the Gaussian noise
% applied to that azimuth position information in the neural network,
% forcing it to encode information in azimuth and azimuth changes around
% the center

% fit curve
x = [0.0100    0.7854    1.5708]; % ~0, 45, 90 degrees azimuth
y = [0.0175    0.0873    0.2618]; % 1, 5, 15 degrees of loc error
f= @(B,x) exp(x).*B(1) + B(2); % function to fit
b=nlinfit(x,y,f,[0.1, -0.5]); % linear fit [0.0647, 0.0506]

% plot fit
xsmooth = x(1)-0.1:0.001:x(3)+0.1;
figure;  plot(xsmooth, f(b, xsmooth))
hold on; scatter(x,y, 100, 'rX');

% plot noise
figure;
npos = -pi/2:pi/32:pi/2;
for np = npos
    % 1/4 of loc error, calculating with 4 stdev spread of noise
    stdev = f(b, abs(np)) / 4;
    nx = np-6*stdev:0.001:np+6*stdev;
    hold on; plot(nx, normpdf(nx, np, stdev));
end
xlabel('Azimuth (rad)')
ylabel('Probability Density')
xticks([-pi/2,-pi/4,0,pi/4,pi/2])
xticklabels({'-\pi/2','-\pi/4','0','\pi/4','\pi/2'})
