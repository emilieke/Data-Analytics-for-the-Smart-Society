function [results_LPC] = SpeakerIdentification_LPC(LPCs1,LPCs2,LPCs3,LPCs4,LPCs5,LPCs6,LPCs7,LPCs8,LPCs9,LPCs10,LPCs11,LPCs12,LPCs13,LPCs14,LPCs15,LPCs16)

results_LPC = [];

for i=8:32
    N_GMM = i;
    % Build GMM models with N_GMM components fitted to training data for each
    % speaker
    options = statset('MaxIter', 500);
    GMModel1 = fitgmdist(LPCs1,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal'); 
    GMModel2 = fitgmdist(LPCs2,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel3 = fitgmdist(LPCs3,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel4 = fitgmdist(LPCs4,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel5 = fitgmdist(LPCs5,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel6 = fitgmdist(LPCs6,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel7 = fitgmdist(LPCs7,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel8 = fitgmdist(LPCs8,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel9 = fitgmdist(LPCs9,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel10 = fitgmdist(LPCs10,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel11 = fitgmdist(LPCs11,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel12 = fitgmdist(LPCs12,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel13 = fitgmdist(LPCs13,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel14 = fitgmdist(LPCs14,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel15 = fitgmdist(LPCs15,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');
    GMModel16 = fitgmdist(LPCs16,N_GMM,'Options',options, 'Replicates', 3, 'CovarianceType', 'diagonal');

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
        [LPCt1,~] = melfcc(Xtest_1, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
        % Transpose matrix
        LPCt1 = LPCt1.';

        % Get log likelihood values
        p1 = [sum(log(pdf(GMModel1, LPCt1))), sum(log(pdf(GMModel2, LPCt1))), sum(log(pdf(GMModel3, LPCt1))),...
        sum(log(pdf(GMModel4, LPCt1))), sum(log(pdf(GMModel5, LPCt1))), sum(log(pdf(GMModel6, LPCt1))),...
        sum(log(pdf(GMModel7, LPCt1))), sum(log(pdf(GMModel8, LPCt1))), sum(log(pdf(GMModel9, LPCt1))),...
        sum(log(pdf(GMModel10, LPCt1))), sum(log(pdf(GMModel11, LPCt1))), sum(log(pdf(GMModel12, LPCt1))),...
        sum(log(pdf(GMModel13, LPCt1))), sum(log(pdf(GMModel14, LPCt1))), sum(log(pdf(GMModel15, LPCt1))),...
        sum(log(pdf(GMModel16, LPCt1)))]; 

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
        [LPCt2,~] = melfcc(Xtest_2, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
        % Transpose matrix
        LPCt2 = LPCt2.';

        % Get log likelihood values
        p2 = [sum(log(pdf(GMModel1, LPCt2))), sum(log(pdf(GMModel2, LPCt2))), sum(log(pdf(GMModel3, LPCt2))),...
        sum(log(pdf(GMModel4, LPCt2))), sum(log(pdf(GMModel5, LPCt2))), sum(log(pdf(GMModel6, LPCt2))),...
        sum(log(pdf(GMModel7, LPCt2))), sum(log(pdf(GMModel8, LPCt2))), sum(log(pdf(GMModel9, LPCt2))),...
        sum(log(pdf(GMModel10, LPCt2))), sum(log(pdf(GMModel11, LPCt2))),sum(log( pdf(GMModel12, LPCt2))),...
        sum(log(pdf(GMModel13, LPCt2))), sum(log(pdf(GMModel14, LPCt2))), sum(log(pdf(GMModel15, LPCt2))),...
        sum(log(pdf(GMModel16, LPCt2)))]; 

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
    results_LPC = [results_LPC;accuracy];
end