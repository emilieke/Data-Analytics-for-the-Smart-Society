clear all; close all; clc;

% -------------------------------------------------------------------------------
% Load training files and extract features
% -------------------------------------------------------------------------------

num_spk = 16;               % Number of speakers
MFCCs = cell(1,num_spk);    % Cell array with features for all speakers
LPCs = cell(1,num_spk);
MFCC_LPCs = cell(1,num_spk);

for i=1:num_spk
    % Create train set for each speaker
    Xtrain = load_train_data('list_train.txt', i);
    % Extract features for each speaker using MFCC
    sr = 16000;
    modelorder = 0;
    [MFCC,~] = melfcc(Xtrain, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
    % Extract features for each speaker using LPC
    modelorder = 8;
    [LPC,~] = melfcc(Xtrain, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
    % Transpose matrix
    MFCC = MFCC.';
    LPC = LPC.';
    % Combine the features
    MFCC_LPC = [MFCC,LPC];
    % Store features for speaker i
    MFCCs{1,i} = MFCC;
    LPCs{1,i} = LPC;
    MFCC_LPCs{1,i} = MFCC_LPC;
end

% -------------------------------------------------------------------------------
% Load  test files
% -------------------------------------------------------------------------------

% Test 1
fid = fopen('list_test1.txt');
info_speech_1 = textscan(fid, '%s%f');
numfich_1 = length(info_speech_1{1}); % total number of test files
fclose(fid);

% Test 2
fid = fopen('list_test2.txt');
info_speech_2 = textscan(fid, '%s%f');
numfich_2 = length(info_speech_2{1}); % total number of test files
fclose(fid);

% -------------------------------------------------------------------------------
% Build GMM models
% -------------------------------------------------------------------------------

% Build GMM models with N_GMM components fitted to training data
results_MFCC = [];
results_LPC = [];
results_MFCC_LPC = [];

% Save all the confusion matrices
cms_MFCC_1=[];
cms_LPC_1=[];
cms_MFCC_LPC_1=[];
cms_MFCC_2=[];
cms_LPC_2=[];
cms_MFCC_LPC_2=[];

for i=8:32
    N_GMM = i;
    GMModels_MFCC = cell(num_spk,1);
    GMModels_LPC = cell(num_spk,1);
    GMModels_MFCC_LPC = cell(num_spk,1);
    for j=1:num_spk
        options = statset('MaxIter', 500);
        GMModels_MFCC{j} = fitgmdist(cell2mat(MFCCs(1,j)),N_GMM,'Options',...
            options,'Replicates', 3, 'CovarianceType', 'diagonal');
        GMModels_LPC{j} = fitgmdist(cell2mat(LPCs(1,j)),N_GMM,'Options',...
            options,'Replicates', 3, 'CovarianceType', 'diagonal');
        GMModels_MFCC_LPC{j} = fitgmdist(cell2mat(MFCC_LPCs(1,j)),N_GMM,...
            'Options',options,'Replicates', 3, 'CovarianceType', 'diagonal');
    end
    
% -------------------------------------------------------------------------------    
% Testing each of the test files with the models    
% -------------------------------------------------------------------------------

% Test 1

    % Create empty arrays for predicted and actual labels
    pred_1_MFCC = [];
    pred_1_LPC = [];   
    pred_1_MFCC_LPC = [];
    
    test_1 = [];
    
    for n=1:numfich_1
        % Load test file
        [Xtest_1,ytest_1] = load_test_data('list_test1.txt',n);
        % Extract features
        modelorder = 0;
        [MFCCt1,~] = melfcc(Xtest_1, sr, 'numcep', 20,'wintime', 0.02,'hoptime',...
        0.01);
        modelorder = 8;
        [LPCt1,~] = melfcc(Xtest_1, sr, 'numcep', 20,'wintime', 0.02,'hoptime',...
        0.01);
        % Transpose matrix
        MFCCt1 = MFCCt1.';
        LPCt1 = LPCt1.';
        % Combine features
        MFCC_LPC_t1 = [MFCCt1,LPCt1];
        % Get log likelihood values
        p1_MFCC = [];
        p1_LPC = [];
        p1_MFCC_LPC = [];
        for j=1:num_spk
            p_MFCC = sum(log(pdf(GMModels_MFCC{j}, MFCCt1)));
            p1_MFCC = [p1_MFCC;p_MFCC];
            
            p_LPC = sum(log(pdf(GMModels_LPC{j}, LPCt1)));
            p1_LPC = [p1_LPC;p_LPC];
            
            p_MFCC_LPC = sum(log(pdf(GMModels_MFCC_LPC{j}, MFCC_LPC_t1)));
            p1_MFCC_LPC = [p1_MFCC_LPC;p_MFCC_LPC];
        end

        % Identify speaker: Get index given by max loglikelihood value
        [~, pred_id_1_MFCC] = max(p1_MFCC);
        pred_1_MFCC = [pred_1_MFCC;pred_id_1_MFCC];
        
        [~, pred_id_1_LPC] = max(p1_LPC);
        pred_1_LPC = [pred_1_LPC;pred_id_1_LPC];
        
        [~, pred_id_1_MFCC_LPC] = max(p1_MFCC_LPC);
        pred_1_MFCC_LPC = [pred_1_MFCC_LPC;pred_id_1_MFCC_LPC];
        
        test_1 = [test_1;ytest_1];
    end
    pred_1_MFCC = int16(pred_1_MFCC); % convert to int array
    pred_1_LPC = int16(pred_1_LPC); % convert to int array
    pred_1_MFCC_LPC = int16(pred_1_MFCC_LPC); % convert to int array
    
% -------------------------------------------------------------------------------
% Test 2

    % Create empty arrays for predicted and actual labels
    pred_2_MFCC = [];
    pred_2_LPC = [];
    pred_2_MFCC_LPC = [];
    
    test_2 = [];
    
    for n=1:numfich_2
        % Load test file
        [Xtest_2,ytest_2] = load_test_data('list_test2.txt',n);
        % Extract features
        modelorder = 0;
        [MFCCt2,~] = melfcc(Xtest_2, sr, 'numcep', 20,'wintime', 0.02,'hoptime',...
            0.01);
        modelorder = 8;
        [LPCt2,~] = melfcc(Xtest_2, sr, 'numcep', 20,'wintime', 0.02,'hoptime',...
            0.01);
        
        % Transpose matrix
        MFCCt2 = MFCCt2.';
        LPCt2 = LPCt2.';
        % Combine features
        MFCC_LPC_t2 = [MFCCt2,LPCt2];
        
        % Get log likelihood values
        p2_MFCC = [];
        p2_LPC = [];
        p2_MFCC_LPC = [];
        
        for j=1:num_spk
            p_MFCC = sum(log(pdf(GMModels_MFCC{j}, MFCCt2)));
            p2_MFCC = [p2_MFCC;p_MFCC];
            
            p_LPC = sum(log(pdf(GMModels_LPC{j}, LPCt2)));
            p2_LPC = [p2_LPC;p_LPC];
            
            p_MFCC_LPC = sum(log(pdf(GMModels_MFCC_LPC{j}, MFCC_LPC_t2)));
            p2_MFCC_LPC = [p2_MFCC_LPC;p_MFCC_LPC];
        end

        % Identify speaker: Get index given by max loglikelihood value
        [~, pred_id_2_MFCC] = max(p2_MFCC);
        pred_2_MFCC = [pred_2_MFCC;pred_id_2_MFCC];
        
        [~, pred_id_2_LPC] = max(p2_LPC);
        pred_2_LPC = [pred_2_LPC;pred_id_2_LPC];
        
        [~, pred_id_2_MFCC_LPC] = max(p2_MFCC_LPC);
        pred_2_MFCC_LPC = [pred_2_MFCC_LPC;pred_id_2_MFCC_LPC];
        
        test_2 = [test_2;ytest_2];
    end
    pred_2_MFCC = int16(pred_2_MFCC); % Convert to int array
    pred_2_LPC = int16(pred_2_LPC); % Convert to int array
    pred_2_MFCC_LPC = int16(pred_2_MFCC_LPC); % Convert to int array
    
% -------------------------------------------------------------------------------    
% Evaluation of the baseline system
% -------------------------------------------------------------------------------    
    
    % Test 1: Confusion matrix
    cm_1_MFCC = confusionmat(test_1,pred_1_MFCC);
    cm_1_LPC = confusionmat(test_1,pred_1_LPC);
    cm_1_MFCC_LPC = confusionmat(test_1,pred_1_MFCC_LPC);
    
    cms_MFCC_1=[cms_MFCC_1;cm_1_MFCC];
    cms_LPC_1=[cms_LPC_1;cm_1_LPC];
    cms_MFCC_LPC_1=[cms_MFCC_LPC_1;cm_1_MFCC_LPC];

    % Accuracy: compute accuracy score based on confusion matrix
    accuracy_1_MFCC = sum(diag(cm_1_MFCC))/sum(cm_1_MFCC(:));
    accuracy_1_LPC = sum(diag(cm_1_LPC))/sum(cm_1_LPC(:));
    accuracy_1_MFCC_LPC = sum(diag(cm_1_MFCC_LPC))/sum(cm_1_MFCC_LPC(:));

    % Test 2: Confusion matrix
    cm_2_MFCC = confusionmat(test_2,pred_2_MFCC);
    cm_2_LPC = confusionmat(test_2,pred_2_LPC);
    cm_2_MFCC_LPC = confusionmat(test_2,pred_2_MFCC_LPC);
    
    cms_MFCC_2=[cms_MFCC_2;cm_2_MFCC];
    cms_LPC_2=[cms_LPC_2;cm_2_LPC];
    cms_MFCC_LPC_2=[cms_MFCC_LPC_2;cm_2_MFCC_LPC];
    
    % Accuracy: compute accuracy score based on confusion matrix
    accuracy_2_MFCC = sum(diag(cm_2_MFCC))/sum(cm_2_MFCC(:));
    accuracy_2_LPC = sum(diag(cm_2_LPC))/sum(cm_2_LPC(:));
    accuracy_2_MFCC_LPC = sum(diag(cm_2_MFCC_LPC))/sum(cm_2_MFCC_LPC(:));
    
    % Get results for different number of GMM components
    accuracy_MFCC = [N_GMM,accuracy_1_MFCC,accuracy_2_MFCC];
    results_MFCC = [results_MFCC;accuracy_MFCC];
    
    accuracy_LPC = [N_GMM,accuracy_1_LPC,accuracy_2_LPC];
    results_LPC = [results_LPC;accuracy_LPC];
    
    accuracy_MFCC_LPC = [N_GMM,accuracy_1_MFCC_LPC,accuracy_2_MFCC_LPC];
    results_MFCC_LPC = [results_MFCC_LPC;accuracy_MFCC_LPC];
    
end

% -------------------------------------------------------------------------------    
% Plot the accuracy
% -------------------------------------------------------------------------------    

figure
plot(results_MFCC(:,1),results_MFCC(:,2),results_LPC(:,1),results_LPC(:,2),...
    results_MFCC_LPC(:,1),results_MFCC_LPC(:,2),results_MFCC(:,1),...
    results_MFCC(:,3),results_LPC(:,1),results_LPC(:,3),results_MFCC_LPC(:,1),...
    results_MFCC_LPC(:,3),'LineWidth', 2)
% Set legend
legend('MFCC clean','LPC clean','MFCC and LPC clean','MFCC noisy','LPC noisy',...
    'MFCC and LPC noisy','Location','SouthEast')
% Set the axis limits
axis([8 32 0.4 1.1])
% Add title and axis labels
xlabel('Number of Gaussians')
ylabel('Accuracy')
title('Accuracy for all models')

% -------------------------------------------------------------------------------    
% Plot the accuracy clean conditions
% -------------------------------------------------------------------------------    

figure
plot(results_MFCC(:,1),results_MFCC(:,2),results_LPC(:,1),results_LPC(:,2),...
    results_MFCC_LPC(:,1),results_MFCC_LPC(:,2),'LineWidth', 2)
% Set legend
legend('MFCC clean','LPC clean','MFCC and LPC clean','Location','SouthEast')
% Set the axis limits
axis([8 32 0.95 1.01])
% Add title and axis labels
xlabel('Number of Gaussians')
ylabel('Accuracy')
title('Accuracy in clean conditions')

% -------------------------------------------------------------------------------    
% Plot the accuracy noisy conditions
% -------------------------------------------------------------------------------    

figure
plot(results_MFCC(:,1),results_MFCC(:,3),results_LPC(:,1),results_LPC(:,3),...
    results_MFCC_LPC(:,1),results_MFCC_LPC(:,3),'LineWidth', 2)
% Set legend
legend('MFCC noisy','LPC noisy','MFCC and LPC noisy','Location','SouthEast')
% Set the axis limits
axis([8 32 0.55 0.77])
% Add title and axis labels
xlabel('Number of Gaussians')
ylabel('Accuracy')
title('Accuracy in noisy conditions')

% -------------------------------------------------------------------------------    