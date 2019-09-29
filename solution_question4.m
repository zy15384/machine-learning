clc;clear all;
x=(-5:0.1:5);
fx1 = (2*pi)^(-0.5)*exp(-0.5*x.^2);
fx2 = (2*pi^(0.5))^(-1)*exp(-0.25*(x-1).^2);
figure (1);
plot(x,fx1,x,fx2),title('the class-conditional pdfs'), xlabel('x'),ylabel('p(x/w)'),legend('p(x/w1)','p(x/w2)')

fx3=fx1./(fx1+fx2);
fx4=fx2./(fx1+fx2);
figure (2);
plot(x,fx3,x,fx4),title('class posterior probabilitys'), xlabel('x'), ylabel('p(w/x)'),legend('p(w1/x)','p(w2/x)')