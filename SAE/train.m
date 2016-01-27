
%% CS294A/CS294W Programming Assignment Starter Code
clc; close all; clear all;
addpath ../common/;
load trainData

patchSize = 28;
visibleSize = patchSize*patchSize;   % number of input units 
hiddenSize = 128;     % number of hidden units 
sparsityParam = 0.035;   % desired average activation of the hidden units. (default 0.001)
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 1e-4; % Weight decay parameter
beta = 3;            % weight of sparsity penalty term  (default 3) 

% patches = sampleIMAGES(data_train, patchSize, 50000);
% load patches
%figure(4);
%display_network(patches(:,100:200),8);

% patches = patches(:,1:1000);
%  Obtain random parameters theta


%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).
    
%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
sparsityParam = 0.001 ;
for i = 1:1
    sparsityParam = sparsityParam + 0.005;
    [opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                        visibleSize, hiddenSize, ...
                                        lambda, sparsityParam, ...
                                        beta, data_train'), ...
                                   theta, options);


    fprintf('Final Cost = %f', cost);
    [cost, grad] = sparseAutoencoderCost(opttheta, visibleSize, hiddenSize, lambda, ...
                                        sparsityParam, beta, data_train');

    save ('pretrained.mat', 'opttheta');
    %%======================================================================
    %% STEP 5: Visualization 

    W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    
    display_network(W1',hiddenSize); 
%     filename = int2str(i) + '.jpg';
%     filename = sprintf('weight_%d_%f.jpg', i,sparsityParam);
%     print('-djpeg', filename);   % save the visualization to a file 
end

% figure(5);
% display_network(W1',hiddenSize); 




