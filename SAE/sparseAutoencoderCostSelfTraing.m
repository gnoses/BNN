function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)


% ---------------------------------------
% debugging initilization
% 
if (1)
clc; close all; clear all;
visibleSize = 64;   % number of input units 
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.001;
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term  
theta = initializeParameters(hiddenSize, visibleSize);
elementSize = 50;

data = rand(visibleSize,elementSize);

end
% debugging initilization
% ---------------------------------------



W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);

b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

learningRate = 1;%0.01;

%elementSize = 100;%length(data); % 100;
CostTerm = 0;

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
a2Table = zeros(elementSize,hiddenSize);
a3 = zeros(1,visibleSize);

rho = zeros(1,hiddenSize);


costHistory = [];

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
for i=1:1:1000000000
    
    % forward pass
    for elementIndex=1:elementSize
        a2Table(elementIndex,:) = sigmoid(W1 * data(:,elementIndex) + b1)';  
        %rho = rho + a2Table(elementIndex,:);
    end

    %rho = rho ./ elementSize;
    
    for elementIndex=1:elementSize
        a2 = a2Table(elementIndex,:);
        a3 = sigmoid(W2 * a2' + b2)';        

        % backward propogation

        % delta of level 3  -(y-a3)*f'(z3)
        d3 = -(data(:,elementIndex)' - a3(1,:)) .* a3(1,:) .* (1-a3(1,:));    
        d2 = (W2' * d3(1,:)' .* a2' .* (1-a2'));

        % sparsity delta 
        %sparsityDelta = beta .* ((-sparsityParam ./ rho) + (1 - sparsityParam) ./ (1-rho));
        %d2 = ((W2' * d3(1,:)' + sparsityDelta') .* a2' .* (1-a2'))';

        % Gradient weight vector cumulation
        W2grad =  W2grad + d3' * a2;
        W1grad = W1grad + d2 * data(:,elementIndex)';

        b1grad = b1grad + d2;
        b2grad = b2grad + d3';


        %elementIndex
        curData =  data(:,elementIndex);
        temp = (a3(:) - curData);

        % draw network
        if (0)
            if (elementIndex <= 5)
                figure (1);
            subplot(5,1,elementIndex), plot(a3, 'r');
            hold on
            subplot(5,1,elementIndex), plot(curData, 'g');
            hold off;
            
            end
        end
        
        CostTerm = CostTerm + 0.5 .* norm(temp,2);
    end

    CostTerm = CostTerm / elementSize;

    %W2grad = learningRate .* (W2grad ./ elementSize);
    %W1grad = learningRate .* (W1grad ./ elementSize);
    W2grad = learningRate .* (W2grad ./ elementSize + W2 .* lambda );
    W1grad = learningRate .* (W1grad ./ elementSize + W1 .* lambda );
    b1grad = learningRate .* (b1grad ./ elementSize);
    b2grad = learningRate .* (b2grad ./ elementSize);

    W1 = W1 - W1grad;
    W2 = W2 - W2grad;
    b1 = b1 - b1grad;
    b2 = b2 - b2grad;

    %Error = sqrt(sum(CostTerm))/elementSize;
    WeightDecay = lambda*(sum(W1(:).*W1(:)) + sum(W2(:).*W2(:)))/2;

    


    SparsityTerm = sum(sparsityParam .* log (sparsityParam ./ rho) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1-rho)));


    cost = CostTerm + WeightDecay ;%+ (beta *SparsityTerm);
    
    if (mod(i,1000) == 0)
        costHistory = [costHistory,cost];
        win1 = figure(2);
        set(win1, 'position', [1000 700 300 300]);
        
        plot(costHistory)
        
        fprintf('#%d cost = %f, (%f, %f)\n', i, cost,CostTerm, WeightDecay)
        
        
        for elementIndex=1:5
            a2 = a2Table(elementIndex,:);
            a3 = sigmoid(W2 * a2' + b2)';  
            curData =  data(:,elementIndex);
            win2 = figure (1);
            set(win2, 'position', [1000 300 300 300]);
            subplot(5,1,elementIndex), plot(a3, 'r');
            hold on
            subplot(5,1,elementIndex), plot(curData, 'g');
            hold off;
        end        
        
        drawnow;
    end
end
 
fprintf('cost = %f\n', cost)

for elementIndex=1:elementSize

    % forward pass

    a2 = sigmoid(W1 * data(:,elementIndex) + b1)';  
    a3 = sigmoid(W2 * a2(1,:)' + b2)';        


    %elementIndex
    curData =  data(:,elementIndex);
    temp = (a3(:) - curData);


    subplot(5,1,elementIndex), plot(a3, 'b');
    hold on
    subplot(5,1,elementIndex), plot(curData, 'r');
    hold off;

end
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
%data(1:10)

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

