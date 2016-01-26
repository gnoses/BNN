function run_train()

clc; clear; close all;


addpath '../common';
addpath(genpath('../common/minFunc_2012/minFunc'));

% [data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();
load trainData


% basic neural network parameter
bnnModel.NNParam = [];
bnnModel.NNParam.input_dim = 784;
bnnModel.NNParam.output_dim = 10;
bnnModel.NNParam.layer_sizes = [256, bnnModel.NNParam.output_dim];
bnnModel.NNParam.lambda = 0;

bnnModel.NNParam.activation_fun = 'tanh';
bnnModel.NNParam.verboseTraining = false;
bnnModel.NNParam.useGPU = false;

% Boosted NN parameter (common, not changed further)
bnnModel.fullFeatureSize = size(data_train,2);
bnnModel.sampledFeatureSize = bnnModel.fullFeatureSize / 2;
bnnModel.NNParam.input_dim = bnnModel.sampledFeatureSize;

if (0)
    tic()    
    nnWeight = TrainNNFull(bnnModel.NNParam, data_train, labels_train);
    fullTrainingTime = toc();
    save ('fullOptTheta', 'nnWeight', 'fullTrainingTime');    
else
    load fullOptTheta;
end

[acc_train, acc_test] = TestFullNN(nnWeight,bnnModel.NNParam, data_train, labels_train, data_test, labels_test);

fprintf('Full MLP train: %f, test: %f ( %f sec)\n', acc_train, acc_test, fullTrainingTime);

% save ()
bnnModelList = {};
bnnModel.NNParam.layer_sizes = [128, bnnModel.NNParam.output_dim];

for t=1:1
    bnnModel.trees = 5+(3*t);
    
%     m = 5000;
    
%     tic()
    batchRatio = [0.5];
    for j=1:size(batchRatio)
        bnnModel.batchRatio = batchRatio(j);
        bnnModel.batchSize = size(data_train,1) * (1 - bnnModel.batchRatio);
                
        bnnModel = TrainBoostedNN(bnnModel, data_train,labels_train,data_test, labels_test);
        
    %     toc()
        [bnnModel.acc_train, bnnModel.acc_test] = TestBoostedNN(bnnModel, data_train, labels_train, data_test, labels_test);
        bnnModelList{end+1} = bnnModel;
        fprintf('Boosted MLP trees:%d, batchsize:%d, train: %f, test: %f\n', bnnModel.trees, bnnModel.batchSize, bnnModel.acc_train, bnnModel.acc_test);
    end    
end

save ('bnnModel', 'bnnModelList');

end



function nnWeight = TrainNNFull(NNParam, data_train, labels_train)
options = [];

options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.Display = 'off';

NNParam.input_dim = size(data_train,2);

stack = initialize_weights(NNParam);
params = stack2params(stack);
% input size must be set as featur dimension
NNParam.input_dim = size(data_train,2);

[nnWeight,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,NNParam, data_train, labels_train);

end



function bnnModelOut = TrainBoostedNN(bnnModelIn, data_train,labels_train,data_test, labels_test)
bnnModelOut = bnnModelIn;
for treeId=1:bnnModelOut.trees
%         index = randperm(60000,m);        
    [data_trainPart, labels_trainPart] = SplitData(data_train, labels_train, bnnModelOut.batchRatio);
    % data_train : m x n
    featureIndex = randperm(bnnModelOut.fullFeatureSize);
    featureIndex = featureIndex(1:bnnModelOut.sampledFeatureSize);
    
    bnnModelOut.nnWeightList{treeId} = TrainNNTrees(treeId,featureIndex,bnnModelOut.NNParam,data_trainPart, labels_trainPart, data_test, labels_test);
    
    bnnModelOut.featureIndexList{treeId} = featureIndex;
end
end

function nnWeight = TrainNNTrees(treeId,featureIndex,NNParam, data_train, labels_train, data_test, labels_test)


options = [];

options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.Display = 'off';

data_trainSampledFeature = data_train(:, featureIndex);
data_test_SampledFeature = data_test(:, featureIndex);



stack = initialize_weights(NNParam);
params = stack2params(stack);

[nnWeight,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,NNParam, data_trainSampledFeature, labels_train);


if (NNParam.verboseTraining)
    %% compute accuracy on the test and train set
    [~, ~, pred] = supervised_dnn_cost( nnWeight, bnnModel.NNParam, data_test_SampledFeature, [], true);
    [~,pred] = max(pred);
    acc_test = mean(pred'==labels_test);
    % fprintf('test accuracy: %f\n', acc_test);

    [~, ~, pred] = supervised_dnn_cost( nnWeight, bnnModel.NNParam, data_train_SampledFeature, [], true);
    [~,pred] = max(pred);
    acc_train = mean(pred'==labels_train);
    fprintf('#%d train : %f, test : %f\n',treeId, acc_train, acc_test);
end
 

end


%nnWeight = params;
% load('nnWeight');
function [acc_train, acc_test] = TestFullNN(nnWeight, NNParam, data_train, labels_train, data_test, labels_test)
predTrainAll = zeros(10, size(labels_train,1));
predTestAll = zeros(10,size(labels_test,1));

theta =  nnWeight;
NNParam.input_dim = size(data_train,2);
[~, ~, pred] = supervised_dnn_cost( theta, NNParam, data_test, [], true);
[score,pred] = max(pred);
evaluation = pred'==labels_test;
acc_test = mean(pred'==labels_test);
desc = sprintf('Full acc = %f', acc_test);
ShowROCCurve(evaluation, score',desc); 

[~, ~, pred] = supervised_dnn_cost( theta, NNParam, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
end

function [acc_train, acc_test] = TestBoostedNN(bnnModel, data_train, labels_train, data_test, labels_test)
predTrainAll = zeros(10, size(labels_train,1));
predTestAll = zeros(10,size(labels_test,1));

for i=1:bnnModel.trees
    nnWeight =  bnnModel.nnWeightList{i};
    featureIndex = bnnModel.featureIndexList{i};
    
    [~, ~, pred] = supervised_dnn_cost( nnWeight, bnnModel.NNParam, data_test(:,featureIndex), [], true);
    predTestAll = predTestAll + pred;
    
    % fprintf('test accuracy: %f\n', acc_test);

    [~, ~, pred] = supervised_dnn_cost( nnWeight, bnnModel.NNParam, data_train(:,featureIndex), [], true);
    predTrainAll = predTrainAll + pred;    
end


pred = predTestAll/bnnModel.trees;
[score,pred] = max(pred);
evaluation = pred'==labels_test;
acc_test = mean(pred'==labels_test);
desc = sprintf('trees = %d, acc = %f', bnnModel.trees, acc_test);
ShowROCCurve(evaluation, score',desc); 
[~,pred] = max(predTrainAll/bnnModel.trees);
acc_train = mean(pred'==labels_train);


end

% label : m x 1, score : m x 1
function ShowROCCurve(label, score, desc)
persistent descList;
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(label,score,'true');
hold on
plot(Xsvm,Ysvm)
xlabel('False positive rate'); ylabel('True positive rate');
descList{end+1} = desc;
legend(descList)
drawnow;
end