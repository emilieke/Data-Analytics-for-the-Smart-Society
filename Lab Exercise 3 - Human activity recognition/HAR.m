% -------------------------------------------------------------------------
% Human Activity Recognition
% Lab Exercise 3
% Emilie Engen, 100356077
% -------------------------------------------------------------------------
clear all; close all; clc;

% -------------------------------------------------------------------------
% Classes
% -------------------------------------------------------------------------
n_classes = 5;
classes = {'Running' 'Walking' 'Standing' 'Sitting' 'Lying'};

% -------------------------------------------------------------------------
% Get training and test sets
% -------------------------------------------------------------------------
% Load preprocessed data
filename = 'HAR_database.mat';
load (filename);

% Data and labels
data = database_training(:,1);
labels = database_training(:,2);

% Convert cell arrays to matrices
data_matrix=[];
labels_matrix=[];
data_test=[];

for i=1:length(labels)
    data_T = data{i,1}.';
    data_matrix = [data_matrix;data_T];
    labels_T = labels{i,1}.';
    labels_matrix = [labels_matrix;labels_T];
end

for i=1:length(database_test)
    data_test_T = database_test{i,1}.';
    data_test = [data_test;data_test_T];
end

% -------------------------------------------------------------------------
% Plot the data
% -------------------------------------------------------------------------
% Plot the data
class_colors = {'red','blue','green','cyan','yellow'};

figure();
for k = 1:5
    subplot(2,3,k);
    t = sprintf('Predictor: %d ',k);
    title(t);
    hold on;
    for i = 1:n_classes
        index = find(labels_matrix==i);
        temp = data_matrix(index,k);
        h = hist(temp,50)/length(temp);
        plot(h,class_colors{i},'LineWidth',3);
    end
    axis tight;
    legend(classes);
end

% -------------------------------------------------------------------------
% Train and evaluate classification models
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% 1. Naive Bayes
% -------------------------------------------------------------------------
% Build Naive Bayes classifier
NB_model = fitcnb(data_matrix,labels_matrix);

% 10-fold CV
NB_model_CV = crossval(NB_model);

% Predict labels
NB_labels = kfoldPredict(NB_model_CV);

% Construct confusion matrix
NB_cm = confusionmat(labels_matrix,NB_labels);

% Inspect dataset
[n,p] = size(data_matrix);
is_labels = unique(labels_matrix);
n_labels = numel(is_labels);
tabulate(categorical(labels_matrix))

% Convert the integer label vector to a class-identifier matrix
[~,group_labels_NB] = ismember(NB_labels,is_labels); 
CI_matrix_NB = zeros(n_labels,n); 
idx_linear_NB = sub2ind([n_labels n],group_labels_NB,(1:n)'); 

% Flags the row corresponding to the class 
CI_matrix_NB(idx_linear_NB) = 1;
[~,groups_NB] = ismember(labels_matrix,is_labels); 
l_matrix_NB = zeros(n_labels,n); 
idx_linear_label_NB = sub2ind([n_labels n],groups_NB,(1:n)'); 
l_matrix_NB(idx_linear_label_NB) = 1; 

% Plot confusion matrix
figure;
plotconfusion(l_matrix_NB,CI_matrix_NB);

% -------------------------------------------------------------------------
% 2. SVM
% -------------------------------------------------------------------------
% Build SVM model
t = templateSVM('Standardize',true);
SVM_model = fitcecoc(data_matrix,labels_matrix,'Learners',t);

% 10-fold CV
SVM_model_CV = crossval(SVM_model);

% Predict labels
SVM_labels = kfoldPredict(SVM_model_CV);

% Confusion matrix
SVM_cm = confusionmat(labels_matrix,SVM_labels);

% Convert the integer label vector to a class-identifier matrix.
[~,group_labels_SVM] = ismember(SVM_labels,is_labels); 
CI_matrix_SVM = zeros(n_labels,n); 
idx_linear_SVM = sub2ind([n_labels n],group_labels_SVM,(1:n)'); 

% Flags the row corresponding to the class 
CI_matrix_SVM(idx_linear_SVM) = 1;
[~,groups_SVM] = ismember(labels_matrix,is_labels); 
l_matrix_SVM = zeros(n_labels,n); 
idx_linear_label_SVM = sub2ind([n_labels n],groups_SVM,(1:n)'); 
l_matrix_SVM(idx_linear_label_SVM) = 1; 

% Plot confusion matrix
figure;
plotconfusion(l_matrix_SVM,CI_matrix_SVM);

% -------------------------------------------------------------------------
% 3. Neural Networks
% -------------------------------------------------------------------------
% Transform labels to binary vectors
labels_bin = zeros(size(labels_matrix,1),5);
for i = 1:size(data_matrix,1)
    if (labels_matrix(i) == 1)
        labels_bin(i,:) = [1 0 0 0 0];
    end
    if (labels_matrix(i) == 2)
        labels_bin(i,:) = [0 1 0 0 0];
    end
    if (labels_matrix(i) == 3)
        labels_bin(i,:) = [0 0 1 0 0];
    end
    if (labels_matrix(i) == 4)
        labels_bin(i,:) = [0 0 0 1 0];
    end
    if (labels_matrix(i) == 5)
        labels_bin(i,:) = [0 0 0 0 1];
    end
end

% Transpose data and labels 
data_matrix_T = data_matrix.';
labels_bin_T = labels_bin.';

% Build Neural Network
NN_model = patternnet([75,50]);
view(NN_model)

% Train Neural Network model
[NN_model,tr] = train(NN_model,data_matrix_T,labels_bin_T);
view(NN_model)

% Predict labels
NN_labels = NN_model(data_matrix_T);

% Get classes
NN_classes = vec2ind(NN_labels);

% Confusion matrix
NN_cm = confusionmat(labels_matrix,NN_classes);

% Convert the integer label vector to a class-identifier matrix.
[~,group_labels_NN] = ismember(NN_classes.',is_labels); 
CI_matrix_NN = zeros(n_labels,n); 
idx_linear_NN = sub2ind([n_labels n],group_labels_NN,(1:n)'); 

% Flags the row corresponding to the class 
CI_matrix_NN(idx_linear_NN) = 1;
[~,groups_NN] = ismember(labels_matrix,is_labels); 
l_matrix_NN = zeros(n_labels,n); 
idx_linear_label_NN = sub2ind([n_labels n],groups_NN,(1:n)'); 
l_matrix_NN(idx_linear_label_NN) = 1; 

% Plot confusion matrix
figure;
plotconfusion(l_matrix_NN,CI_matrix_NN);

% -------------------------------------------------------------------------
% Predict activity for test set
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% 1. Naive Bayes
% -------------------------------------------------------------------------
% Predict label for test set
NB_labels_test = predict(NB_model,data_test);

% Insert labels in cell array
NB_labels_test = NB_labels_test.';
len_test = length(database_test{1,1});
db_NB_labels_test = cell(2,[]);
db_NB_labels_test{1,1} = NB_labels_test(1:len_test);
db_NB_labels_test{2,1} = NB_labels_test(len_test+1:length(NB_labels_test));

% Save results
save('NB_results.mat','db_NB_labels_test')

% -------------------------------------------------------------------------
% 2. SVM
% -------------------------------------------------------------------------
% Predict label for test set
SVM_labels_test = predict(SVM_model,data_test);

% Insert labels in cell array
SVM_labels_test = SVM_labels_test.';
db_SVM_labels_test = cell(2,[]);
db_SVM_labels_test{1,1} = SVM_labels_test(1:len_test);
db_SVM_labels_test{2,1} = SVM_labels_test(len_test+1:length(SVM_labels_test));

% Save results
save('SVM_results.mat','db_SVM_labels_test')

% -------------------------------------------------------------------------
% 3. Neural network
% -------------------------------------------------------------------------
% Transpose test data
data_test_T = data_test.';

% Predict labels
NN_labels_test = NN_model(data_test_T);

% Get classes
NN_classes_test = vec2ind(NN_labels_test);

% Insert labels in cell array
db_NN_labels_test = cell(2,[]);
db_NN_labels_test{1,1} = NN_classes_test(1:len_test);
db_NN_labels_test{2,1} = NN_classes_test(len_test+1:length(NN_classes_test));

% Save results
save('NN_results.mat','db_NN_labels_test')

% -------------------------------------------------------------------------