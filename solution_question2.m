clc;
clear all;
x=(-5:0.1:5);
fx=2*exp(-abs(x)+0.5*abs(x-1));
plot(x,fx,'r'), xlabel('x'),ylabel('l(x)'),title('log-likelihood-ratio function')