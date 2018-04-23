clear all; close all; clc;

% --------------------------------------------------------------------
% Image Classes
% --------------------------------------------------------------------

Num_img_classes = 4;
Classes = categorical({'cartman' 'cowboy-hat' 'galaxy' 'hibiscus'});

% --------------------------------------------------------------------
% TRAINING AND TEST SETS
% --------------------------------------------------------------------

TrSet_size = 61;
TestSet_size = 20;
Num_features_per_image = 1131;
%Num_features_per_image = 1200;

% --------------------------------------------------------------------
% FEATURE EXTRACTION
% --------------------------------------------------------------------

% Note: In order for the HOG function to run I had to include the 
% following code lines after declaring the variable bin, to make sure
% bin was no greater than 8, which would give an index out of bound
% error:

% if bin>8
%   bin=8;
% end

% --------------------------------------------------------------------
% Get Xtrain
% --------------------------------------------------------------------

% Obtain images from directory
dir_train = 'Train_Set'; 
files_train = dir([dir_train, '/*.jpg']);

% Total number of images in the traning set
N_Images_train = length(files_train); 

Xtrain=[];
for k=1:N_Images_train
    % Read image
    M = imread([dir_train '/' files_train(k).name]);
    % Convert to double
    M = im2double(M);
    
    % Resize image
    h = 320; % height 
    l = 240; % lenght
    M = imresize(M,[h l]);
    [rows columns num_Col_Channels] = size(M);
    
    % Convert RGB images
    if num_Col_Channels==3
        M = rgb2gray(M);
    end
    
    % Feature extraction
    x_cellnum=2;        % Split block in 2x2 cells
    y_cellnum=2;
    num_grad_or=8;      % Number of gradient orientation
    b_size = 16;        % block size (16x16)
    step = b_size/2;    % step length for creating blocks (50% overlap)
    
    h_max=h-b_size+1;   % starting point for last block in image (y direction)
    l_max=l-b_size+1;   % starting point for last block in image (x direction)
    count=0;
    for i=1:step:h_max
        for j=1:step:l_max
            block = M(i:i+b_size-1, j:j+b_size-1); % create block
            count=count+1; % count number of blocks per image
            H = HOG(block,x_cellnum, y_cellnum, num_grad_or); % extract feature from block
            H = H.'; % transpose H
            Xtrain = [Xtrain; H]; % add histogram for each block in Xtrain
        end
    end
end

% --------------------------------------------------------------------
% Get Xtest
% --------------------------------------------------------------------

% Obtain images from directory
dir_test = 'Test_Set'; 
files_test = dir([dir_test, '/*.jpg']);

% Total number of images in the traning set
N_Images_test = length(files_test); 

Xtest=[];
for k=1:N_Images_test
    % Read image
    M = imread([dir_test '/' files_test(k).name]);
    % Convert to double
    M = im2double(M);
    
    % Resize image
    M = imresize(M,[h l]);
    [rows columns num_Col_Channels] = size(M);
    
    % Convert RGB images to grayscale
    if num_Col_Channels==3
        M = rgb2gray(M);
    end
    
    % Feature extraction
    for i=1:step:h_max
        for j=1:step:l_max
            block = M(i:i+b_size-1, j:j+b_size-1); % create block
            H = HOG(block,x_cellnum, y_cellnum, num_grad_or); % extract feature from block
            H = H.'; % transpose H
            Xtest = [Xtest; H]; % add histogram for each block in Xtest
        end
    end
end

%load Xtrain;
%load Xtest;

% --------------------------------------------------------------------
% CREATING A VISUAL VOCABULARY
% --------------------------------------------------------------------

Vocabulary_Size = 100;
%load vocabulario_K100;

% Create vocabulary from Xtrain
[Cind C] = kmeans(Xtrain,Vocabulary_Size);

% Memory allocation for Histograms of visual words
Hist = zeros(Num_img_classes,TrSet_size,Vocabulary_Size);  

% --------------------------------------------------------------------
% Computing histograms
% --------------------------------------------------------------------

for i=1:Num_img_classes
    for j=1:TrSet_size
        
        % Visual words asignation
        image = (i-1)*TrSet_size+j;                 % Image number
        i1 = (image-1)*Num_features_per_image+1;    % First block in image
        i2 = i1 + Num_features_per_image -1;        % Last block in image
        Cind_k = knnsearch(C,Xtrain(i1:i2,:));      % Get closest word in vocabulary
        
        % histogram computation
        H = hist(Cind_k,Vocabulary_Size);
        
        % Histogram normalization (sum=1)
        H_normalized = H/sum(H); 
        
        % Insert normalized value in histogram matrix
        for w=1:Vocabulary_Size
            Hist(i,j,w) = H_normalized(w);
        end 
    end
end


% --------------------------------------------------------------------
% TRAINING AN IMAGE CATEGORY CLASSIFIER
% --------------------------------------------------------------------

% Label Vector
Ytrain=[];
for i=1:Num_img_classes
    for j=1:TrSet_size
        Ytrain = [Ytrain; Classes(i)]; % Add true label
    end
end

% Reshaped Histogram Matrix
H_Xtrain=[];
for i=1:Num_img_classes
    for j=1:TrSet_size
        H_Xtrain = [H_Xtrain; reshape(Hist(i,j,:), 1, Vocabulary_Size)];
    end
end

% Train classifier
t = templateSVM('Standardize',true);
Classifier= fitcecoc(H_Xtrain,Ytrain,'Learners',t);

% --------------------------------------------------------------------
% Performance on the Training Set
% --------------------------------------------------------------------

% Predicted_Y: predict labels for images in train set
Predicted_Y = predict(Classifier,H_Xtrain);

% confMatrix: create confusion matrix
confMatrix = confusionmat(Ytrain,Predicted_Y);

% accuracy: compute accuracy score based on confusion matrix
accuracy = sum(diag(confMatrix))/sum(confMatrix(:));


% --------------------------------------------------------------------
% REAL PERFORMANCE: PERFORMANCE ON THE TEST SET
% --------------------------------------------------------------------

% Memory allocation for Histograms of visual words
Hist_Test = zeros(Num_img_classes,TestSet_size,Vocabulary_Size);  

% --------------------------------------------------------------------
% Computing histograms
% --------------------------------------------------------------------


for i=1:Num_img_classes
    for j=1:TestSet_size
        
        % Visual words asignation
        image = (i-1)*TestSet_size+j;               % Image number
        i1 = (image-1)*Num_features_per_image+1;    % First block in image
        i2 = i1 + Num_features_per_image -1;        % Last block in image
        Cind_k_Test = knnsearch(C,Xtest(i1:i2,:));  % Get closest word in vocabulary
        
        % histogram computation
        H_test = hist(Cind_k_Test,Vocabulary_Size);
        
        % Histogram normalization (sum=1)
        H_normalized_Test = H_test/sum(H_test); 
        
        % Insert normalized value in histogram matrix
        for w=1:Vocabulary_Size
            Hist_Test(i,j,w) = H_normalized_Test(w);
        end      
    end
end


% --------------------------------------------------------------------
% Evaluation
% --------------------------------------------------------------------

% Label Vector: Y_Test
Y_Test=[];
for i=1:Num_img_classes
    for j=1:TestSet_size
        Y_Test = [Y_Test; Classes(i)]; % Add true label
    end
end

% Reshaped Histogram Matrix: H_X_Test
H_X_Test=[];
for i=1:Num_img_classes
    for j=1:TestSet_size
        H_X_Test = [H_X_Test; reshape(Hist_Test(i,j,:),1,Vocabulary_Size)];
    end
end

% Predicted_Y_Test: predict labels for images in test set
Predicted_Y_Test = predict(Classifier,H_X_Test);

% confMatrix_Test: create confusion matrix
confMatrix_Test = confusionmat(Y_Test,Predicted_Y_Test);

% accuracy_Test: compute accuracy score based on confusion matrix
accuracy_Test = sum(diag(confMatrix_Test))/sum(confMatrix_Test(:));

