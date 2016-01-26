function [Xpart, Ypart] = SplitData(X, Y, samplingThr)

% data_train : [ featureDim x m ]
% labels_train : [ 1 x m ]

labels = unique(Y);
% X = X';
% Y = Y';
% samplingThr = 0.4; % percentage for testing data sampling
Xpart = [];
Ypart = [];


for i=1:length(labels)
    singleClassDataX = X(Y==labels(i),:);
    singleClassDataY = Y(Y==labels(i));
   

    selected = rand(size(singleClassDataX,1),1);
    Xpart = cat(1,Xpart, singleClassDataX(selected >= samplingThr,:));
  
    
    Ypart = cat(1,Ypart, singleClassDataY(selected >= samplingThr));
  
end

% Xpart = Xpart';
% Ypart = Ypart';
% X2part = X2part';
% Y2part = Y2part';
% X2=X2part;
% Y2=Y2part;
end