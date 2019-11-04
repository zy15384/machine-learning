clear;close all;


% tolerance for EM stopping criterion
delta = 1;
% at beginning, the delta is set to 0.5.
% however, the running time of the for loop takes
% too much time, it the delta is not large enough.
% in this case, i set delta with the value of 1 finally.

regWeight = 1e-10; % regularization parameter for covariance estimates

% Generate samples from a 4-component GMM
N=100;
K=10;
alpha_true = [0.4,0.3,0.2,0.1];
mu_true = [1 10 -10 0;0 0 0 0];
Sigma_true(:,:,1) = [3 1;1 8];
Sigma_true(:,:,2) = [2 1;1 2];
Sigma_true(:,:,3) = [8 1;1 16];
Sigma_true(:,:,4) = [15 1;1 12];
x = randGMM(N,alpha_true,mu_true,Sigma_true);
[labels,data] = x;

[label_10,data_10] = generateTrueGMM(10);
[label_100,data_100] = generateTrueGMM(100);
[label_1000,data_1000] = generateTrueGMM(1000);
[label_10000,data_10000] = generateTrueGMM(10000);

figure(1),
subplot(2,2,1)
plot(data_10(1,:),data_10(2,:),'g.');
title('True data of 10 samples');
axis equal,
subplot(2,2,2)
plot(data_100(1,:),data_100(2,:),'b.');
title('True data of 100 samples');
axis equal,
subplot(2,2,3)
plot(data_1000(1,:),data_1000(2,:),'r.');
title('True data of 1000 samples');
axis equal,
subplot(2,2,4)
plot(data_10000(1,:),data_10000(2,:),'k.');
title('True data of 10000 samples');
axis equal,

[d,L] = size(mu_true); % determine dimensionality of samples and number of GMM components

% Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

% Allocate space
MSEtrain = zeros(K,N); MSEvalidate = zeros(K,N); 
AverageMSEtrain = zeros(1,N); AverageMSEvalidate = zeros(1,N);

% Try all polynomial orders between 1 (best line fit) and N-1 (big time overfit)

% an empty matrix p is used to save the performace of 6 Gaussian components
p = [];
for M = 1:6
    % K-fold cross validation
    for k = 1:K
     [xValidate,xTrain,Ntrain,Nvalidate] = KFold(indPartitionLimits,K,k,N);   
   % Initialize the GMM to randomly selected samples
    alpha = ones(1,M)/M;
    shuffledIndices = randperm(Ntrain);
    mu = xTrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
    %mu = x(:,shuffledIndices(1:M));
     [~,assignedCentroidLabels] = min(pdist2(mu',xTrain'),[],1); % assign each sample to the nearest mean
 %   [~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1);
    for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(xTrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
    end 
    t = 0; %displayProgress(t,x,alpha,mu,Sigma);
    
    Converged = 0; % Not converged at the beginning
    kk = 1;
    while ~Converged
        for l = 1:M
            %temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
            temp(l,:) = repmat(alpha(l),1,Ntrain).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
%         w = plgivenx./repmat(sum(plgivenx,2),1,N);
        w = plgivenx./repmat(sum(plgivenx,2),1,Ntrain);
        muNew = xTrain*w';
        for l = 1:M
%           v = xTrain-repmat(muNew(:,l),1,N);
            v = xTrain-repmat(muNew(:,l),1,Ntrain);
            u = repmat(w(l,:),d,1).*v;
            SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha'));
        Dmu = sum(sum(abs(muNew-mu)));
        DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
        Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
        t = t+1; 
        %displayProgress(t,xTrain,alpha,mu,Sigma);
    end
        %p=perform(M,alpha,mu,Sigma,xValidate,k);
        
    end
    s = perform(M,alpha,mu,Sigma,xValidate,k);
    p=[p,s(M)];
    
end
 p
% here shows the function of performance calculation
function p=perform(M,alpha,mu,Sigma,xValidate,k)
perf=zeros(1,10);
p=zeros(1,6);

 for i=1:10
    for j=1:M
        perf(i)=perf(i)+alpha(j)*mvnpdf(xValidate(:,i),mu(:,j),Sigma(:,:,j));
        
    end
    
 end
  for j=1:M
          p(M)=(1/k)*sum(log(perf(j)));
          
  end
end

% this is the function of kFold application
function [xValidate,xTrain,Ntrain,Nvalidate] = KFold(indPartitionLimits,K,k,N)
         
        x = 10*randn(2,N);
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        
        xValidate = x(:,indValidate); % Using folk k as validation set
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            %indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
            indTrain = [[1:indPartitionLimits(k-1,2)],[indPartitionLimits(k+1,1):N]];
            
        end
         %xTrain = x(indTrain);
         xTrain = x(:,indTrain); % using all other folds as training set
         Ntrain = length(indTrain); Nvalidate = length(indValidate); 
end

function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(0.1),
end

function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end


function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
%figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
% E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma.*(x-repmat(mu,1,N))),1);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end