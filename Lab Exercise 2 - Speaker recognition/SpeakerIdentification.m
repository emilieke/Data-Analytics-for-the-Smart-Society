clear all; close all; clc;

% --------------------------------------------------------------------
% Loading of the training files
% --------------------------------------------------------------------

% Train data

% Create train set for each speaker
Xtrain_s1 = load_train_data('list_train.txt', 1);
Xtrain_s2 = load_train_data('list_train.txt', 2);
Xtrain_s3 = load_train_data('list_train.txt', 3);
Xtrain_s4 = load_train_data('list_train.txt', 4);
Xtrain_s5 = load_train_data('list_train.txt', 5);
Xtrain_s6 = load_train_data('list_train.txt', 6);
Xtrain_s7 = load_train_data('list_train.txt', 7);
Xtrain_s8 = load_train_data('list_train.txt', 8);
Xtrain_s9 = load_train_data('list_train.txt', 9);
Xtrain_s10 = load_train_data('list_train.txt', 10);
Xtrain_s11 = load_train_data('list_train.txt', 11);
Xtrain_s12 = load_train_data('list_train.txt', 12);
Xtrain_s13 = load_train_data('list_train.txt', 13);
Xtrain_s14 = load_train_data('list_train.txt', 14);
Xtrain_s15 = load_train_data('list_train.txt', 15);
Xtrain_s16 = load_train_data('list_train.txt', 16);

% --------------------------------------------------------------------
% Feature extraction
% --------------------------------------------------------------------

% Extract features for each speaker using MFCC
sr = 16000;
[MFCCs1,~] = melfcc(Xtrain_s1, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs2,~] = melfcc(Xtrain_s2, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs3,~] = melfcc(Xtrain_s3, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs4,~] = melfcc(Xtrain_s4, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs5,~] = melfcc(Xtrain_s5, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs6,~] = melfcc(Xtrain_s6, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs7,~] = melfcc(Xtrain_s7, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs8,~] = melfcc(Xtrain_s8, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs9,~] = melfcc(Xtrain_s9, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs10,~] = melfcc(Xtrain_s10, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs11,~] = melfcc(Xtrain_s11, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs12,~] = melfcc(Xtrain_s12, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs13,~] = melfcc(Xtrain_s13, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs14,~] = melfcc(Xtrain_s14, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs15,~] = melfcc(Xtrain_s15, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[MFCCs16,~] = melfcc(Xtrain_s16, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);

MFCCs1 = MFCCs1.';
MFCCs2 = MFCCs2.';
MFCCs3 = MFCCs3.';
MFCCs4 = MFCCs4.';
MFCCs5 = MFCCs5.';
MFCCs6 = MFCCs6.';
MFCCs7 = MFCCs7.';
MFCCs8 = MFCCs8.';
MFCCs9 = MFCCs9.';
MFCCs10 = MFCCs10.';
MFCCs11 = MFCCs11.';
MFCCs12 = MFCCs12.';
MFCCs13 = MFCCs13.';
MFCCs14 = MFCCs14.';
MFCCs15 = MFCCs15.';
MFCCs16 = MFCCs16.';

% Extract features for each speaker using LPC
modelorder = 8;
[LPCs1,~] = melfcc(Xtrain_s1, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs2,~] = melfcc(Xtrain_s2, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs3,~] = melfcc(Xtrain_s3, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs4,~] = melfcc(Xtrain_s4, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs5,~] = melfcc(Xtrain_s5, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs6,~] = melfcc(Xtrain_s6, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs7,~] = melfcc(Xtrain_s7, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs8,~] = melfcc(Xtrain_s8, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs9,~] = melfcc(Xtrain_s9, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs10,~] = melfcc(Xtrain_s10, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs11,~] = melfcc(Xtrain_s11, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs12,~] = melfcc(Xtrain_s12, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs13,~] = melfcc(Xtrain_s13, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs14,~] = melfcc(Xtrain_s14, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs15,~] = melfcc(Xtrain_s15, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
[LPCs16,~] = melfcc(Xtrain_s16, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);

LPCs1 = LPCs1.';
LPCs2 = LPCs2.';
LPCs3 = LPCs3.';
LPCs4 = LPCs4.';
LPCs5 = LPCs5.';
LPCs6 = LPCs6.';
LPCs7 = LPCs7.';
LPCs8 = LPCs8.';
LPCs9 = LPCs9.';
LPCs10 = LPCs10.';
LPCs11 = LPCs11.';
LPCs12 = LPCs12.';
LPCs13 = LPCs13.';
LPCs14 = LPCs14.';
LPCs15 = LPCs15.';
LPCs16 = LPCs16.';

% Combine both features
Xs1 = [MFCCs1,LPCs1];
Xs2 = [MFCCs2,LPCs2];
Xs3 = [MFCCs3,LPCs3];
Xs4 = [MFCCs4,LPCs4];
Xs5 = [MFCCs5,LPCs5];
Xs6 = [MFCCs6,LPCs6];
Xs7 = [MFCCs7,LPCs7];
Xs8 = [MFCCs8,LPCs8];
Xs9 = [MFCCs9,LPCs9];
Xs10 = [MFCCs10,LPCs10];
Xs11 = [MFCCs11,LPCs11];
Xs12 = [MFCCs12,LPCs12];
Xs13 = [MFCCs13,LPCs13];
Xs14 = [MFCCs14,LPCs14];
Xs15 = [MFCCs15,LPCs15];
Xs16 = [MFCCs16,LPCs16];

% --------------------------------------------------------------------
% Speaker GMM models building
% --------------------------------------------------------------------

results_MFCC = [];
%results_LPC = [];
%results_MFCC_LPC = [];

for i=8:32
    N_GMM = i;
    % Build GMM models with N_GMM components fitted to training data for each
    % speaker
    options = statset('MaxIter', 500);
    GMModel1 = fitgmdist(MFCCs1,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal'); 
    GMModel2 = fitgmdist(MFCCs2,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel3 = fitgmdist(MFCCs3,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel4 = fitgmdist(MFCCs4,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel5 = fitgmdist(MFCCs5,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel6 = fitgmdist(MFCCs6,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel7 = fitgmdist(MFCCs7,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel8 = fitgmdist(MFCCs8,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel9 = fitgmdist(MFCCs9,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel10 = fitgmdist(MFCCs10,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel11 = fitgmdist(MFCCs11,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel12 = fitgmdist(MFCCs12,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel13 = fitgmdist(MFCCs13,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel14 = fitgmdist(MFCCs14,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel15 = fitgmdist(MFCCs15,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel16 = fitgmdist(MFCCs16,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');

    % --------------------------------------------------------------------
    % Testing each of the test files with the models
    % --------------------------------------------------------------------

    % Test data 1
    fid = fopen('list_test1.txt');
    info_speech_1 = textscan(fid, '%s%f');
    numfich_1 = length(info_speech_1{1}); % total number of test files
    fclose(fid);

    % Create empty arrays for predicted and actual labels
    pred_1 = [];
    test_1_id_list = [];
    for j=1:numfich_1
        % Load test file
        [Xtest_1,ytest_1] = load_test_data('list_test1.txt',j);
        % Extract features
        [MFCCt1,~] = melfcc(Xtest_1, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
        % Transpose matrix
        MFCCt1 = MFCCt1.';

        % Get log likelihood values
        p1 = [sum(log(pdf(GMModel1, MFCCt1))), sum(log(pdf(GMModel2, MFCCt1))), sum(log(pdf(GMModel3, MFCCt1))),...
        sum(log(pdf(GMModel4, MFCCt1))), sum(log(pdf(GMModel5, MFCCt1))), sum(log(pdf(GMModel6, MFCCt1))),...
        sum(log(pdf(GMModel7, MFCCt1))), sum(log(pdf(GMModel8, MFCCt1))), sum(log(pdf(GMModel9, MFCCt1))),...
        sum(log(pdf(GMModel10, MFCCt1))), sum(log(pdf(GMModel11, MFCCt1))), sum(log(pdf(GMModel12, MFCCt1))),...
        sum(log(pdf(GMModel13, MFCCt1))), sum(log(pdf(GMModel14, MFCCt1))), sum(log(pdf(GMModel15, MFCCt1))),...
        sum(log(pdf(GMModel16, MFCCt1)))]; 

        % Identify speaker: Get index given by max loglikelihood value
        [~, pred_id_1] = max(p1);
        pred_1 = [pred_1;pred_id_1];
        test_1_id_list = [test_1_id_list;ytest_1];
    end
    pred_1 = int16(pred_1); % convert to int array

    % --------------------------------------------------------------------

    % Test data 2
    fid = fopen('list_test2.txt');
    info_speech_2 = textscan(fid, '%s%f');
    numfich_2 = length(info_speech_2{1}); % total number of test files
    fclose(fid);

    % Create empty arrays for predicted and actual labels
    pred_2 = [];
    test_2_id_list = [];
    for i=1:numfich_2
        % Load test file
        [Xtest_2,ytest_2] = load_test_data('list_test2.txt',i);
        % Extract features
        [MFCCt2,~] = melfcc(Xtest_2, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
        % Transpose matrix
        MFCCt2 = MFCCt2.';

        % Get log likelihood values
        p2 = [sum(log(pdf(GMModel1, MFCCt2))), sum(log(pdf(GMModel2, MFCCt2))), sum(log(pdf(GMModel3, MFCCt2))),...
        sum(log(pdf(GMModel4, MFCCt2))), sum(log(pdf(GMModel5, MFCCt2))), sum(log(pdf(GMModel6, MFCCt2))),...
        sum(log(pdf(GMModel7, MFCCt2))), sum(log(pdf(GMModel8, MFCCt2))), sum(log(pdf(GMModel9, MFCCt2))),...
        sum(log(pdf(GMModel10, MFCCt2))), sum(log(pdf(GMModel11, MFCCt2))),sum(log( pdf(GMModel12, MFCCt2))),...
        sum(log(pdf(GMModel13, MFCCt2))), sum(log(pdf(GMModel14, MFCCt2))), sum(log(pdf(GMModel15, MFCCt2))),...
        sum(log(pdf(GMModel16, MFCCt2)))]; 

        % Identify speaker: Get index given by max loglikelihood value
        [~, pred_id_2] = max(p2);
        pred_2 = [pred_2;pred_id_2];
        test_2_id_list = [test_2_id_list;ytest_2];
    end
    pred_2 = int16(pred_2); % convert to int array

    % --------------------------------------------------------------------
    % Evaluation of the baseline system
    % --------------------------------------------------------------------
    % Test 1: measure identification accuracy

    % Confusion matrix
    confMatrix_test_1 = confusionmat(test_1_id_list,pred_1);

    % Identification accuracy
    accuracy_test_1 = sum(diag(confMatrix_test_1))/sum(confMatrix_test_1(:));

    % --------------------------------------------------------------------
    % Test 2: measure identification accuracy

    % Confusion matrix
    confMatrix_test_2 = confusionmat(test_2_id_list,pred_2);

    % accuracy_Test: compute accuracy score based on confusion matrix
    accuracy_test_2 = sum(diag(confMatrix_test_2))/sum(confMatrix_test_2(:));

    % --------------------------------------------------------------------

    accuracy = [N_GMM,accuracy_test_1,accuracy_test_2];
    results_MFCC = [results_MFCC;accuracy];
end



