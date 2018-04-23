function [results_MFCC_LPC] = SpeakerIdentification_MFCC_LPC(Xs1,Xs2,Xs3,Xs4,Xs5,Xs6,Xs7,Xs8,Xs9,Xs10,Xs11,Xs12,Xs13,Xs14,Xs15,Xs16)

results_MFCC_LPC = [];

for i=8:32
    N_GMM = i;
    % Build GMM models with N_GMM components fitted to training data for each
    % speaker
    options = statset('MaxIter', 500);
    GMModel1 = fitgmdist(Xs1,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal'); 
    GMModel2 = fitgmdist(Xs2,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel3 = fitgmdist(Xs3,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel4 = fitgmdist(Xs4,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel5 = fitgmdist(Xs5,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel6 = fitgmdist(Xs6,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel7 = fitgmdist(Xs7,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel8 = fitgmdist(Xs8,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel9 = fitgmdist(Xs9,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel10 = fitgmdist(Xs10,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel11 = fitgmdist(Xs11,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel12 = fitgmdist(Xs12,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel13 = fitgmdist(Xs13,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel14 = fitgmdist(Xs14,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel15 = fitgmdist(Xs15,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel16 = fitgmdist(Xs16,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');

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
    for i=1:numfich_1
        % Load test file
        [Xtest_1,ytest_1] = load_test_data('list_test1.txt',i);
        % Extract features
        [Xt1,~] = melfcc(Xtest_1, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
        % Transpose matrix
        Xt1 = Xt1.';

        % Get log likelihood values
        p1 = [sum(log(pdf(GMModel1, Xt1))), sum(log(pdf(GMModel2, Xt1))), sum(log(pdf(GMModel3, Xt1))),...
        sum(log(pdf(GMModel4, Xt1))), sum(log(pdf(GMModel5, Xt1))), sum(log(pdf(GMModel6, Xt1))),...
        sum(log(pdf(GMModel7, Xt1))), sum(log(pdf(GMModel8, Xt1))), sum(log(pdf(GMModel9, Xt1))),...
        sum(log(pdf(GMModel10, Xt1))), sum(log(pdf(GMModel11, Xt1))), sum(log(pdf(GMModel12, Xt1))),...
        sum(log(pdf(GMModel13, Xt1))), sum(log(pdf(GMModel14, Xt1))), sum(log(pdf(GMModel15, Xt1))),...
        sum(log(pdf(GMModel16, Xt1)))]; 

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
        [Xt2,~] = melfcc(Xtest_2, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
        % Transpose matrix
        Xt2 = Xt2.';

        % Get log likelihood values
        p2 = [sum(log(pdf(GMModel1, Xt2))), sum(log(pdf(GMModel2, Xt2))), sum(log(pdf(GMModel3, Xt2))),...
        sum(log(pdf(GMModel4, Xt2))), sum(log(pdf(GMModel5, Xt2))), sum(log(pdf(GMModel6, Xt2))),...
        sum(log(pdf(GMModel7, Xt2))), sum(log(pdf(GMModel8, Xt2))), sum(log(pdf(GMModel9, Xt2))),...
        sum(log(pdf(GMModel10, Xt2))), sum(log(pdf(GMModel11, Xt2))),sum(log( pdf(GMModel12, Xt2))),...
        sum(log(pdf(GMModel13, Xt2))), sum(log(pdf(GMModel14, Xt2))), sum(log(pdf(GMModel15, Xt2))),...
        sum(log(pdf(GMModel16, Xt2)))]; 

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

    % accuracy = [N_GMM,accuracy_test_1,accuracy_test_2];
    accuracy = [N_GMM,accuracy_test_1,accuracy_test_2];
    results_MFCC_LPC = [results_MFCC_LPC;accuracy];
end