clc; clear all;

% Produce the samples. 
Mu = [-1 0; 1 0;0 1]';% mean
Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % covariance of data pdf conditioned on label 3
Sigma(:,:,2) = 0.1*[5 0;0,2]; % covariance of data pdf conditioned on label 2
Sigma(:,:,3) = 0.1*eye(2); % covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]';

N = 10000;
[X, Y] = generate_gauss_classes(Mu, Sigma, classPriors, N);
 
figure();
hold on;
class1 = X(:, Y==1);
class2 = X(:, Y==2);
class3 = X(:, Y==3);

% Samples were plotted with labels.
plot(class1(1, :), class1(2, :), 'c.');
plot(class2(1, :), class2(2, :), 'm.');
plot(class3(1, :), class3(2, :), 'y.');
grid on;
title('Samples Plot');
xlabel('x');
ylabel('y');
legend('class1','class2','class3');
ax = gca;
ax.FontSize = 15;
axis equal,

% Justify the correct of class label.
class1 = class1';
class2 = class2';
class3 = class3';

right1 = 0;
right2 = 0;
right3 = 0;
r12 = 0;
r13 = 0;
r21 = 0;
r23 = 0;
r31 = 0;
r32 = 0;
% For the class 1. label the error point
for i = 1:(N*classPriors(1))
    x = class1(i,1);
    y = class1(i,2);
  %plot(class1(:,1),class1(:,2),'r+');

P1 = mvnpdf([x,y]', Mu(:,1), Sigma(:, :, 1));
P2 = mvnpdf([x,y]', Mu(:,2), Sigma(:, :, 2));
P3 = mvnpdf([x,y]', Mu(:,3), Sigma(:, :, 3));

if P1 > P2
    if P1 > P3
        %plot(class1(i,1),class1(i,2),'rs');
        hold on;
        right1 = right1 + 1;
    else 
        plot(class1(i,1),class1(i,2),'bs');
        r13 = r13 + 1;
    end
else if P1 < P2
        if P2 < P3   
            plot(class1(i,1),class1(i,2),'bs');
            r13 = r13 + 1;
        else
            plot(class1(i,1),class1(i,2),'gs');
            r12 = r12 + 1;
        end
    hold on;
    end
end
end
errorRate1 = 1 - (right1 / (N*classPriors(1)));

% For the class 2. label the error point
for i = 1:(N*classPriors(2))
    x = class2(i,1);
    y = class2(i,2);
  %plot(class2(:,1),class2(:,2),'g+');

P1 = mvnpdf([x,y]', Mu(:,1), Sigma(:, :, 1));
P2 = mvnpdf([x,y]', Mu(:,2), Sigma(:, :, 2));
P3 = mvnpdf([x,y]', Mu(:,3), Sigma(:, :, 3));

if P1 > P2
    if P1 > P3
        plot(class2(i,1),class2(i,2),'rs');
        r21 = r21 + 1;
    	hold on;
    else
        plot(class2(i,1),class2(i,2),'bs');
        r23 = r23 + 1;
    end
else if P1 < P2
        if P2 > P3
            %plot(class2(i,1),class2(i,2),'gs');
            right2 = right2 + 1;
        else
            plot(class2(i,1),class2(i,2),'bs');
            r23 = r23 + 1;
        end
    hold on;
    end
end
end
errorRate2 = 1 - (right2 / (N*classPriors(2)));

% For the class 3. label the error point
for i = 1:(N*classPriors(3))
    x = class3(i,1);
    y = class3(i,2);
  %plot(class3(:,1),class3(:,2),'b+');

P1 = mvnpdf([x,y]', Mu(:,1), Sigma(:, :, 1));
P2 = mvnpdf([x,y]', Mu(:,2), Sigma(:, :, 2));
P3 = mvnpdf([x,y]', Mu(:,3), Sigma(:, :, 3));

if P3 > P2
    if P1 > P3
        plot(class3(i,1),class3(i,2),'rs');
        r31 = r31 + 1;
    	hold on;
    else
        %plot(class3(i,1),class3(i,2),'bs');
        right3 = right3 + 1;
    end
else if P3 < P2
        if P2 > P1
            plot(class3(i,1),class3(i,2),'gs');
            r32 = r32 + 1;
        else
            plot(class3(i,1),class3(i,2),'rs');
            r31 = r31 + 1;
        end
    hold on;
    end
end
end
errorRate3 = 1 - (right3 / (N*classPriors(3)));


disp(['The actual number for the first class is ', num2str(N*classPriors(1))]);
disp(['The error rate of the first class is calculated as ',num2str(errorRate1*100),'%']);
disp(['The total number of errors for the first class is ', num2str(errorRate1*(N*classPriors(1)))]);

disp(['The actual number for the second class is ', num2str(N*classPriors(2))]);
disp(['The error rate of the second class is calculated as ',num2str(errorRate2*100),'%']);
disp(['The total number of errors for the second class is ', num2str(errorRate2*(N*classPriors(2)))]);

disp(['The actual number for the third class is ', num2str(N*classPriors(3))]);
disp(['The error rate of the third class is calculated as ',num2str(errorRate3*100),'%']);
disp(['The total number of errors for the third class is ', num2str(errorRate3*(N*classPriors(3)))]);


function [ data, C ] = generate_gauss_classes( M, S, P, N )
 
[~, c] = size(M);
data = [];
C = [];
 
for j = 1:c

    t = mvnrnd(M(:,j), S(:,:,j), fix(P(j)*N))';
    
    data = [data t];
    C = [C ones(1, fix(P(j) * N)) * j];
end
end
