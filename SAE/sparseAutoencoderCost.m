function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)


% ---------------------------------------
% debugging initilization

% if (exist('theta') == 0)
%     load('sparseAutoencoderCost');
% end
% debugging initilization
% ---------------------------------------


W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);

b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
%for i=1:1:10000



elementSize = length(data); % 100;

W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
a2Table = zeros(hiddenSize, elementSize);

rho = zeros(1,hiddenSize);


% forward pass
a2Table = sigmoid(bsxfun(@plus, W1 * data, b1));
rho = sum(a2Table,2)' ./ elementSize;
a3Table = sigmoid(bsxfun(@plus, W2 * a2Table, b2));
d3Table = (a3Table - data) .* a3Table .* (1 - a3Table);

sparsityDelta = beta .* (-(sparsityParam ./ rho) + ((1 - sparsityParam) ./ (1-rho)));
d2Table = (bsxfun(@plus, W2' * d3Table , sparsityDelta')) .* a2Table .* (1-a2Table);
W2grad = ((d3Table * a2Table') ./ elementSize) + (W2 * lambda);
W1grad = ((d2Table * data') ./ elementSize) + (W1 * lambda);
b1grad = sum(d2Table, 2) / elementSize;
b2grad = sum(d3Table, 2) / elementSize;
CostTerm = sum((sum((data - a3Table) .* (data - a3Table), 1))) / elementSize / 2;



WeightDecay = lambda*(sum(W1(:).*W1(:)) + sum(W2(:).*W2(:)))/2;
SparsityTerm2 = (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1-rho));
SparsityTerm = beta * sum((sparsityParam .* log (sparsityParam ./ rho)) + SparsityTerm2);
cost = CostTerm + WeightDecay + SparsityTerm;

 
%fprintf('cost = %f\n', cost)

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
%data(1:10)

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

%fprintf('cost = %f, (%f, %f, %f)\n', cost,CostTerm, WeightDecay, SparsityTerm)


end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

