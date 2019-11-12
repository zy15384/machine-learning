clear all, 
close all;

% trainning set generation
N=1000; n = 2; K=10;

mu = zeros(n,1);
Sigma = eye(n);

p = [0.35,0.65]; % class priors for labels 0 and 1 respectively

% Generate samples
label = rand(1,N) >= p(1);
L = 2*label-1;
Nc = [length(find(label==0)),length(find(label==1))]; 
% number of samples from each class
x = zeros(n,N); % reserve space
theta = zeros(1,N);
rad = zeros(1,N);

thetaRange = [-pi,pi];
radRange = [2,3];

% Draw the random samples from each class pdf
x(:,label==0) = randGaussian(Nc(1),mu,Sigma);
rad = rand(1,Nc(2))+2;
theta = rand(1,Nc(2))*2*pi-pi;
x(1,label==1) = rad.*cos(theta);
x(2,label==1) = rad.*sin(theta)+1;


% output the figure with legend and title
figure(1)
plot(x(1,label==0),x(2,label==0),'c.'), hold on
plot(x(1,label==1),x(2,label==1),'m.'), axis equal
xlabel('x1'), ylabel('x2')
legend('class "-" data','class "+" data')
title('training set generation')

%% linear SVM
% Train a Linear kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
% this part is modified from Deniz code
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end
CList = 10.^linspace(-3,7,11);
for CCounter = 1:length(CList)
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        lValidate = L(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k+1,1):N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k-1,2)];
        else
            indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
        end
        % using all other folds as training set
        xTrain = x(:,indTrain); lTrain = L(indTrain);
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
        indCORRECT = find(lValidate.*dValidate == 1); 
        indINCORRECT = find(lValidate.*dValidate == -1);
        Ncorrect(k)=length(indCORRECT);
        Nincorrect(k)=length(indINCORRECT);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
    PIncorrect(CCounter) = sum(Nincorrect)/N;
end 

% plot the training data and linear-SVM cross-val accuracy estimate
figure(2), subplot(1,2,1),
plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
figure(2), subplot(1,2,1),
title('Linear-SVM Cross-Val Accuracy Estimate'),
[minError,indi] = min(PIncorrect);
CBest1= CList(indi); minError
SVMBest = fitcsvm(x',L','BoxConstraint',CBest1,'KernelFunction','linear');
d = SVMBest.predict(x')'; 
% Labels of training data using the trained SVM
indINCORRECT = find(L.*d == -1); 
% Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(L.*d == 1); 
% Find training samples that are correctly classified by the trained SVM
figure(2), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'c.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'m.'), axis equal,
title('Training Data (magenta: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N, 
% Empirical estimate of training error probability
xlabel('x1'), ylabel('x2'), 
axis equal

%% Gaussian SVM
% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
% this code is also modified from Deniz code in google drive
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = L(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k+1,1):N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k-1,2)];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = L(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1);
            indINCORRECT = find(lValidate.*dValidate == -1);
            Nincorrect(k)=length(indINCORRECT);
            Ncorrect(k)=length(indCORRECT);
        end
        PCorrect(CCounter,sigmaCounter)=sum(Ncorrect)/N;
        PIncorrect(CCounter,sigmaCounter)= sum(Nincorrect)/N;
    end 
end

% draw the picture of gaussian svm
figure(3), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[minError,indi] = min(PIncorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PIncorrect),indi);
CBest2= CList(indBestC); sigmaBest2= sigmaList(indBestSigma); minError
SVMBest = fitcsvm(x',L','BoxConstraint',CBest2,'KernelFunction','gaussian','KernelScale',sigmaBest2);
d = SVMBest.predict(x')'; 
% Labels of training data using the trained SVM
indINCORRECT = find(L.*d == -1); 
% Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(L.*d == 1); 
% Find training samples that are correctly classified by the trained SVM
figure(3), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'c.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'m.'), axis equal,
title('Training Data (magenta: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N, 
% Empirical estimate of training error probability
xlabel('x1'), ylabel('x2'), axis equal

%% test
% Generate samples
label = rand(1,N) >= p(1); L = 2*(label-0.5);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % reserve space
theta = zeros(1,N);
rad = zeros(1,N);

% Draw samples from each class pdf
x(:,label==0) = randGaussian(Nc(1),mu,Sigma);
theta(label==1) = thetaRange(1)+(thetaRange(2)-thetaRange(1)).*rand(1,Nc(2));
rad(label==1) = radRange(1)+(radRange(2)-radRange(1)).*rand(1,Nc(2));
x(:,label==1) = [rad(label==1).*cos(theta(label==1)); rad(label==1).*sin(theta(label==1))];

SVMBest = fitcsvm(x',L','BoxConstraint',CBest1,'KernelFunction','linear');
d = SVMBest.predict(x')'; 
% Labels of training data using the trained SVM
indINCORRECT = find(L.*d == -1);
% Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(L.*d == 1); 
% Find training samples that are correctly classified by the trained SVM
figure(4), subplot(1,2,1), 
plot(x(1,indCORRECT),x(2,indCORRECT),'c.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'m.'), axis equal,
title('Testing Data by Linear SVM(magenta: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N, 
% Empirical estimate of training error probability
xlabel('x1'), ylabel('x2'), 
axis equal

SVMBest = fitcsvm(x',L','BoxConstraint',CBest2,'KernelFunction','gaussian','KernelScale',sigmaBest2);
d = SVMBest.predict(x')'; 
% Labels of training data using the trained SVM
indINCORRECT = find(L.*d == -1); 
% Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(L.*d == 1); 
% Find training samples that are correctly classified by the trained SVM

figure(4), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'c.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'m.'), axis equal,
title('Gassian SVM Test(magenta: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N, 
% Empirical estimate of training error probability
xlabel('x1'), ylabel('x2'), axis equal