clear all; clc;
N = 10;
M = 100;
P = 20;
sigma = 2;

L_min = zeros([1,P]);
L_25 = zeros([1,P]);
L_50 = zeros([1,P]);
L_75 = zeros([1,P]);
L_max = zeros([1,P]);

X = zeros([4,N]);
x = zeros([1,N]);



for p = 1:P
gamma = 2*10^(p-10)*eye(4);
% X = zeros([4,N]);
% x = zeros([1,N]);

%assume w
W = [1,-0.15,-0.5,0.15]';


W_save = zeros([4,100]);
L_save = zeros([1,100]);

for a = 1:M

for j = 1:N
    x(j) = 2*(rand()-0.5);
    % x = [x^3, x^2, x, 1]
    X(:,j) = [x(j)^3; x(j)^2; x(j); 1];
end
%%

%the function is
y = zeros([1,N]);
for j = 1:N
    y(j) = W'*X(:,j) + normrnd(0,sigma);
end

sum_X = zeros([4,4]);
sum_yx = zeros([4,1]);


    for j=1:N
      sum_X = sum_X + X(:,j)*X(:,j)';
      sum_yx = sum_yx + y(j)*X(:,j);
    end
  

W_map = inv(sum_X + inv(gamma^2))* sum_yx/sigma;
W_ml = inv(sum_X)* sum_yx/sigma;
L2 = sum((W_map-W).^2);

W_save(:,a) = W_map;
L_save(a) = L2;
end
%%
[maximum,id] = sort(L_save);
% W_min = W_save(:,id(1));
% W_25 = W_save(:,id(0.25*M));
% W_50 = W_save(:,id(50));
% W_75 = W_save(:,id(75));
% W_max = W_save(:,id(100));
l_min = maximum(1);
l_25 = maximum(25);
l_50 = maximum(50);
l_75 = maximum(75);
l_max = maximum(100);

L_min(p) = l_min;
L_25(p) = l_25;
L_50(p) = l_50;
L_75(p) = l_75;
L_max(p) = l_max;

end



%% plot
GAMMA = logspace(-10,10,20);
loglog(GAMMA,L_min,'*',GAMMA,L_min);
hold on
loglog(GAMMA,L_25,'*',GAMMA,L_25);
hold on
loglog(GAMMA,L_50,'*',GAMMA,L_50);
hold on
loglog(GAMMA,L_75,'*',GAMMA,L_75);
hold on
loglog(GAMMA,L_max,'*',GAMMA,L_max);
xlabel('gamma'),ylabel('Squared_Error'),
legend('minimum errors','25% errors','50% errors','75% errors','maximum errors');

% plot(x,y,'r.');
% hold on
% xx=-1:0.01:1;
% yy=polyval(W_map,xx);
% plot(xx,yy)
% %fplot('W_map(1)*x^3 + W_map(2)*x^2 + W_map(3)*x + 1',[-1,1]);
% hold on
% yy=polyval(W_ml,xx);
% plot(xx,yy)
% %fplot('W_ml(1)*x^3 + W_ml(2)*x^2 + W_ml(3)*x + 1',[-1,1]);
% hold on
% 
% legend('data point','MAP estimation','ML estimation');
% hold on
% 
% grid on