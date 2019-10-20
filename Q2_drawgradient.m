function[] = Q2_drawgradient(k, xt, yt, xi, yi,sigma)
sigmax = sigma;
sigmay = sigma;
sigmai(1:k) = sigma;
xx = linspace(-2,2,100);
yy = linspace(-2,2,100);
[x,y] = meshgrid(xx,yy);
for i = 1:k
    ri(i) = sqrt((xi(i) - xt)^2 +(yi(i) - yt)^2) + normrnd(0,sigma);
    if ri(i) < 0
       i = i - 1;
    end
end
ma = -0.5*log(2*pi) - log(sigmai(1)) - ((ri(1) - sqrt((x-xi(1)).^2 +(y - yi(1)).^2)).^2)./(2*sigmai(1)^2);
for i = 2:k
    ma = ma + (-0.5*log(2*pi) - log(sigmai(i)) - ((ri(i) - sqrt((x-xi(i)).^2 +(y - yi(i)).^2)).^2)./(2*sigmai(i)^2));
end
z = ma + (log(2*pi) - log(sigmax) - log(sigmay) - 0.5.*((x.^2)./sigmax^2 + (y.^2)./sigmay^2));
contour(x,y,z);
axis([-2 2 -2 2]);

%grid minor;