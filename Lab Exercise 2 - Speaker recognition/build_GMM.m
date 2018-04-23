function [N_GMM,accuracy_test_1,accuracy_test_2] = build_GMM(N_GMM)
    % Build GMM models with N_GMM components fitted to training data for each
    % speaker
    options = statset('MaxIter', 500);
    GMModel1 = fitgmdist(MFCCs1,N_GMM,'Options',options); 
    GMModel2 = fitgmdist(MFCCs2,N_GMM,'Options',options);
    GMModel3 = fitgmdist(MFCCs3,N_GMM,'Options',options);
    GMModel4 = fitgmdist(MFCCs4,N_GMM,'Options',options);
    GMModel5 = fitgmdist(MFCCs5,N_GMM,'Options',options);
    GMModel6 = fitgmdist(MFCCs6,N_GMM,'Options',options);
    GMModel7 = fitgmdist(MFCCs7,N_GMM,'Options',options);
    GMModel8 = fitgmdist(MFCCs8,N_GMM,'Options',options);
    GMModel9 = fitgmdist(MFCCs9,N_GMM,'Options',options);
    GMModel10 = fitgmdist(MFCCs10,N_GMM,'Options',options);
    GMModel11 = fitgmdist(MFCCs11,N_GMM,'Options',options);
    GMModel12 = fitgmdist(MFCCs12,N_GMM,'Options',options);
    GMModel13 = fitgmdist(MFCCs13,N_GMM,'Options',options);
    GMModel14 = fitgmdist(MFCCs14,N_GMM,'Options',options);
    GMModel15 = fitgmdist(MFCCs15,N_GMM,'Options',options);
    GMModel16 = fitgmdist(MFCCs16,N_GMM,'Options',options);

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
        [MFCCt1,aspc_t1] = melfcc(Xtest_1, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
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
        [MFCCt2,aspc_t2] = melfcc(Xtest_2, sr, 'numcep', 20,'wintime', 0.02,'hoptime', 0.01);
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

end

