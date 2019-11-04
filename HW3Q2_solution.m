%% intriduction of machine learning hw3 Q2
% question 1
% set parameter
clc;
clear all;
N = 999;
p1 = 0.3;
p2 = 0.7;
mu1 = [0;0]; %mean of distribution 1
sigma1 = [1,-0.5;-0.5,1]; %variance of distribution 1
mu2 = [3;3]; %mean of distribution 2
sigma2 = [1,0.7;0.7,1]; %variance of distribution 2
%generate data
data1 = mvnrnd(mu1,sigma1,floor(p1*N)); %generate gaussian data point
data2 = mvnrnd(mu2,sigma2,ceil(p2*N));
data_label = [zeros([size(data1,1),1]);ones([size(data2,1),1])]; %data label
% plot to see
figure()
plot(data1(:,1),data1(:,2),'b.');
hold on
plot(data2(:,1),data2(:,2),'r+');
hold on
legend('data1','data2');
hold on
title('Original data');
grid on;

%% fisher LDA, question 2
Sb = (mu1 - mu2)*(mu1 - mu2)';
Sw = sigma1+sigma2;
A = inv(Sw)*Sb;
[U,S,V] = svd(A); %SVD
w = V(:,1);
k_LDA = -w(1)/w(2);
%%%%%%%%%%%%%%% used to calculate b for LDA boundry
mu1_1 = mu1'*w;
sigma1_1 = w'*sigma1*w;
mu2_1 = mu2'*w;
sigma2_1 = w'*sigma2*w;
a = -0.5*(1/sigma1_1 - 1/sigma2_1);
b = mu1_1/sigma1_1 - mu2_1/sigma2_1;
c = -mu1_1^2/(2*sigma1_1) + mu2_1^2/(2*sigma2_1) - 0.5*log(sigma1_1/sigma2_1);
temp = roots([a,b,c]);
if size(temp,1) == 2
    if (mu1_1 - temp(1))'*(mu2_1 - temp(1)) < 0
        b_LDA = temp(1);
    else
        b_LDA = temp(2);
    end
else
    b_LDA = temp(1);
end
b_LDA = b_LDA/w(2);
%%%%%%%%%%%%%%%%%used to calculate b for LDA boundry

% choose point positive or negative or error
data = [data1;data2];
negative_point = [];
positive_point = [];
error_points = [];
for i = 1:size(data,1)
    if k_LDA*data(i,1) + b_LDA - data(i,2) >= 0
        if data_label(i) == 0
            positive_point = [positive_point;data(i,:)];
        else
            error_points = [error_points;data(i,:)];
        end
    else
        if data_label(i) == 1
            negative_point = [negative_point;data(i,:)];
        else
            error_points = [error_points;data(i,:)];
        end
    end
end
%plot result
a_min = min(data(:,1));
b_max = max(data(:,1));
error = size(error_points,1)/size(data,1);
figure()
plot(positive_point(:,1),positive_point(:,2),'b.');
hold on
plot(negative_point(:,1),negative_point(:,2),'r+');
hold on
if size(error_points,1) ~= 0
    plot(error_points(:,1),error_points(:,2),'go');
    hold on
end
syms x y;
y = k_LDA*x+b_LDA;
fplot(y,[a_min,b_max]);
hold on
if size(error_points,1) ~= 0
    legend('positive point','negative point','error point','fisher LDA boundry');
else
    legend('positive point','negative point','fisher LDA boundry');
end
grid on
title(['fisher LDA boundry, error is: ', num2str(error)]);
%% logistic function question 3
omiga = [0.01,0.01];
b = 1;
yita = 0.005;
use_data = [];

for i = 1:size(data,1)
    use_data = [use_data; data(i,:)];
    omiga = omiga - yita*sum((data_label(1:size(use_data,1))'- 1./(1+exp(omiga*use_data' + b.*ones([1,size(use_data,1)]))))'.*use_data);
    b = b - yita*sum(data_label(1:size(use_data,1))'- 1./(1+exp(omiga*use_data' + b.*ones([1,size(use_data,1)]))));
end
%divide two data
class1 = [];
class2 = [];
error_point2 = [];
for i = 1: size(data,1)
    if 1 - 1/(1+exp(omiga*data(i,:)' + b)) < 0.5
        if data_label(i) == 1
            class1 = [class1; data(i,:)];
        else
            error_point2 = [error_point2;data(i,:)]; 
        end
    else
        if data_label(i) == 0
            class2 = [class2; data(i,:)];
        else
            error_point2 = [error_point2;data(i,:)];
        end
    end
end
% plot result
error2 = size(error_point2,1)/size(data,1);
syms x y;
y = -x*omiga(1)/omiga(2) - b/omiga(2);
figure()
plot(class1(:,1),class1(:,2),'r+');
hold on
plot(class2(:,1),class2(:,2),'b.');
hold on
if size(error_point2,1) ~= 0
    plot(error_point2(:,1),error_point2(:,2),'go');
    hold on
end
fplot(y,[a_min,b_max]);
hold on
if size(error_point2,1) ~= 0
    legend('class 1','class 2','error point: ','boundry');
else
    legend('class 1','class 2','boundry');
end
hold on
title(['logistic result, error is: ',num2str(error2)]);
grid on
%% MAP use true distribution
class1 = [];
class2 = [];
error_point3 = [];
for i = 1:size(data,1)
    if p1*1/(sqrt(2*pi*norm(sigma1)))*exp(-0.5*(data(i,:)'-mu1)'*inv(sigma1)*(data(i,:)'-mu1)) < p2*1/(sqrt(2*pi*norm(sigma2)))*exp(-0.5* (data(i,:)'-mu2)'*inv(sigma2) *(data(i,:)'-mu2))
        if data_label(i) == 1
            class1 = [class1;data(i,:)];
        else
            error_point3 = [error_point3;data(i,:)];
        end
    else
        if data_label(i) == 0
            class2 = [class2;data(i,:)];
        else
            error_point3 = [error_point3;data(i,:)];
        end
    end
end
error3 = size(error_point3,1)/size(data,1);
figure()
plot(class1(:,1),class1(:,2),'r+');
hold on
plot(class2(:,1),class2(:,2),'b.');
hold on
if size(error_point3,1) ~= 0
    plot(error_point3(:,1),error_point3(:,2),'go');
    hold on
end
if size(error_point3,1) ~= 0
    legend('class 1','class 2','error point: ');
else
    legend('class 1','class 2');
end
hold on
title(['MAP result, error is: ',num2str(error3)]);
grid on