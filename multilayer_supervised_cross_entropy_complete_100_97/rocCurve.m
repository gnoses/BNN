clc;
clear;
load ionosphere
label = strcmp(Y,'b'); % resp = 1, if Y = 'b', or 0 if Y = 'g'
pred = X(:,3:34);
mdlSVM = fitcsvm(pred,label,'Standardize',true);
mdlSVM = fitPosterior(mdlSVM);
[~,score_svm] = resubPredict(mdlSVM);

% label : m x 1, score : m x 1
score = score_svm(:,mdlSVM.ClassNames);
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(label,score,'true');

plot(Xsvm,Ysvm)

legend('Logistic Regression','Support Vector Machines','Naive Bayes','Location','Best');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Logistic Regression, SVM, and Naive Bayes Classification');



