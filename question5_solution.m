clc; clear all;
N=100;
n=2;
uz = zeros(2,1);
sigma_z = eye(2,2);
z = mvnrnd(uz,sigma_z,N)
plot(z(:,1),z(:,2),'r+'),xlabel('x'),ylabel('random vector'),title('n-dimentional random vector')

hold on

A = [1 1; 0 1];
b = [5 6];
sigma_x = A.*A';
x = mvnrnd(b,sigma_x,N)
plot(x(:,1),x(:,2),'*')

xz = A * z' + b'.*ones([1,N])
plot(xz(1,:),xz(2,:),'g.'), legend('z','x','az+b')
