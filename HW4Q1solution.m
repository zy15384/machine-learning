clear all;
clc;

delta = 1e-5; % tolerance for k-mean and EM stopping criterion
regWeight = 1e-10; 
% pixel reading
% imread function is used to load the figure information from the same fold
figure = imread('colorPlane.jpg');
%A = imread('colorBird.jpg');
A = im2double(figure);
% then translate into double

% set up 5-dimensional data for K-means clustering
data = zeros(5,N); % x, y, R, G, B
dataClassified = zeros(3,N);
for j = 1:c
    for i = 1:r
        data(1,i+(j-1)*r) = (j-1)/c; 
        data(2,i+(j-1)*r) = (i-1)/r; 
        data(3,i+(j-1)*r) = A(i,j,1); 
        data(4,i+(j-1)*r) = A(i,j,2); 
        data(5,i+(j-1)*r) = A(i,j,3); 
    end
end

[r,c,n] = size(figure);
N = r*c; % number of samples
figureClassified = zeros(r,c,n);



%% k-mean clustering algorithm
% this part is edited from Deniz code in google drive
for K = 2:5
    shuffledIndices = randperm(N);
    mu = data(:,shuffledIndices(1:K)); % pick K random samples as initial mean estimates
    generated_mu = mu;
    [~,assignedCentroidLabels] = min(pdist2(mu',data'),[],1); 
    % assign each sample to the nearest mean
    Converged = 0;
    while ~Converged
        [~,assignedCentroidLabels] = min(pdist2(mu',data'),[],1); 
        % assign each sample to the nearest mean
        for k = 1:K
            generated_mu(:,k)= mean(data(:,assignedCentroidLabels == k),2);
        end
        Dmu = sum(sum(abs(generated_mu-mu)));
        Converged = (Dmu < delta); % Check if converged
        mu = generated_mu;
    end
    for k = 1:K
        dataClassified(:,assignedCentroidLabels == k) = repmat(mu(3:5,k),1,length(find(assignedCentroidLabels == k))); % assign mu value to classified x
    end
    for j = 1:c
        for i = 1:r
            for ii = 1:3
                figureClassified(i,j,ii) = dataClassified(ii,i+(j-1)*r);
            end
        end
    end
    figure(K-1)
    image(figureClassified)
    title(['K-mean clustering algorithm under ',num2str(K),' segments'])
end

%% GMM-based clustering
% this part is also edited from Deniz code in google drive
d = 5;
for K = 2:5
    % GMM
    alpha = ones(1,K)/K;
    shuffledIndices = randperm(N);
    mu = data(:,shuffledIndices(1:K));
    % pick K random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',data'),[],1); 
    % assign each sample to the nearest mean
    
    for k = 1:K 
        Sigma(:,:,k) = cov(data(:,assignedCentroidLabels == k)') + regWeight*eye(d,d);
    end
    Converged = 0;
    temp = zeros(K,N);
    while ~Converged
        for l = 1:K
            temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(data,mu(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
        w = plgivenx./repmat(sum(plgivenx,2),1,N);
        generated_mu = data*w';
        for l = 1:K
            v = data-repmat(generated_mu(:,l),1,N);
            u = repmat(w(l,:),d,1).*v;
            generated_Sigma(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha));
        Dmu = sum(sum(abs(generated_mu-mu)));
        DSigma = sum(sum(sum(abs(generated_Sigma-Sigma))));
        Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        alpha = alphaNew; mu = generated_mu; Sigma = generated_Sigma;
    end
    % MAP
    % for MAP, the code from solution from homework 2 is used here
    evalResult = zeros(K,N);
    for k = 1:K
        evalResult(k,:) = log(evalGaussian(data,mu(:,k),Sigma(:,:,k)))+log(alpha(k));
    end
    [Max,D] = max(evalResult);
    for k = 1:K
        dataClassified(:,D == k) = repmat(mu(3:5,k),1,length(find(D == k))); % assign mu value to classified x
    end
    for j = 1:c
        for i = 1:r
            for ii = 1:3
                figureClassified(i,j,ii) = dataClassified(ii,i+(j-1)*r);
            end
        end
    end
    figure(K+10-1)
    image(figureClassified)
    % same as above,image function is used here to draw the picture
    title([ 'MAP classification using GMM clustering, K is ',num2str(K),''])
end