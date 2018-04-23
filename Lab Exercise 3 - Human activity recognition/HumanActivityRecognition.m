% -------------------------------------------------------------------------------
% Preprocessing
% -------------------------------------------------------------------------------
% Load preprocessed data

filename = 'HAR_database.mat';
load (filename);

% Data and labels
data = database_training(:,1);
target = database_training(:,2);

data_matrix=[];
target_matrix=[];

for i=1:length(target)
    data_T = data{i,1}.';
    data_matrix = [data_matrix;data_T];
    target_T = target{i,1}.';
    target_matrix = [target_matrix;target_T];
end

%Split the data into testing and training:
p = 0.8;
train_length = floor(p*length(target_matrix));
trainData = data_matrix(1:train_length,:);
testData = data_matrix((train_length+1):length(target_matrix),:);
trainLabels =target_matrix(1:train_length,:);
testLabels =target_matrix((train_length+1):length(target_matrix),:);

% -------------------------------------------------------------------------------
% Evaluation
% -------------------------------------------------------------------------------

classifiers = {'knn','NaiveBayes','SVM','SVMlight'};

for iClassifier = classifiers
    method = iClassifier{:};
    classifier = training(trainData, trainLabels, method);
    [scores] = classification(classifier, testData);
end



% -------------------------------------------------------------------------------

%N_classes = 5;
%Classes = {'Running' 'Walking' 'Standing' 'Sitting' 'Lying'};

%N_actors = size(database_training,1);
%trainLabels = database_training(:,2);
%trainData = database_training(:,1);


% Get data
%for i=1:size(N_actors)
%    for j=1:size(trainData{i,1},1) % number of sensor data (5)
%        for k=1:size(trainData{i,1},2) % number of observations
%            data=trainData{i,1}(j,k);
%        end
%    end
%end

% Get label
%for i=1:size(N_actors)
%    for j=1:size(size(trainLabels{i,1},2))
%        label=trainLabels{i,1}(j);
%    end
%end
