function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;

if ~exist('theta','var')
  load('dnn_cost');
end;

if exist('pred_only','var')
  po = pred_only;
end;
% m = size(data,2);
data = (data');
%% reshape into network
if (ei.useGPU)
    data = gpuArray(data);
    % labels = gpuArray(labels');
    theta = gpuArray(theta);
   
end
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hZ = cell(numHidden+2, 1);
hAct = cell(numHidden+2, 1);
gradStack = cell(numHidden+1, 1);

% forward prop
%% YOUR CODE HERE %%%
% tic();
outputLayer = numHidden+2;
hZ{1} = (data);
hAct{1} = (data);
weightSum = 0;
for i=2:outputLayer-1    
%     mul = stack{i-1}.W * hAct{i-1};
    hZ{i} = bsxfun(@plus, stack{i-1}.W * hAct{i-1}, stack{i-1}.b);
%     writeMat(hZ{i}, 'Binary/z1.bin');
    hAct{i} = tanh(hZ{i});
    
    weightSum = weightSum + sum(stack{i-1}.W(:) .^ 2);
end

m = size(data,2);


%% return here if only predictions desired.
% hZ{outputLayer} = stack{outputLayer-1}.W * hAct{outputLayer-1};
hAct{outputLayer} = (stack{outputLayer-1}.W * hAct{outputLayer-1});
weightSum = weightSum + sum(stack{outputLayer-1}.W(:) .^ 2);

h = hAct{outputLayer};

% h = exp(bsxfun(@minus, h, max(h,[], 1)));

h = bsxfun(@rdivide,exp(h) , sum(exp(h),1));



if po
    cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;    
    % pred_prob = gather(pred_prob);
    
    if (ei.useGPU)
        pred_prob = gather(h);
    else
        pred_prob = (h);
    end
    grad = []; 
    return;
end;

if (ei.useGPU)
    groundTruth = gpuArray(full(sparse(labels, 1:m, 1)));
else
    groundTruth = (full(sparse(labels, 1:m, 1)));
end

weightDecay = ei.lambda*(weightSum)/2;
cost = -sum(sum(groundTruth .* log(h))) / m + weightDecay;

d = h - groundTruth;
% tic();
for i=outputLayer:-1:2
    w = stack{i-1}.W;
    gradStack{i-1}.W = d * hAct{i-1}' ./ m + ei.lambda*stack{i-1}.W;
    gradStack{i-1}.b = sum(d, 2) ./ m;

    d = (w' * d) .* (1 - (tanh(hZ{i-1}) .^ 2) ); %hAct{i-1} .* (1 - hAct{i-1});
end
% toc();

if (ei.useGPU)
    cost = gather(cost);
    % grad = gather(grad);
    % pred_prob = gather(pred_prob);
    hAct = gather(hAct);
    %% reshape gradients into vector
    [grad] = gather(stack2params(gradStack));
else
    [grad] = (stack2params(gradStack));
end
end





