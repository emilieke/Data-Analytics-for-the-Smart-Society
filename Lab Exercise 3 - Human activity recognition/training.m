% -------------------------------------------------------------------------------
% Preprocessing
% -------------------------------------------------------------------------------

function classifier = training(trainData, trainLabels, method, varargin)

    if iscell(trainData)
        classifier = cell(length(trainData), 1);
        for t = 1:length(trainData)
             classifier{t} = training(trainData{t}, trainLabels, 'method', method);
        end
    else
        trainData = trainData(:);

        switch method
            case 'NaiveBayes'           
                classifier.model = NaiveBayes.fit(trainData, trainLabels, options.training{:}); % GMM
                classifier.posterior = @naiveBayes;

            case 'knn'
                classifier.model.traindata = trainData;
                classifier.model.labels = trainLabels;
                classifier.posterior = @knn;

            case 'SVM'
                classifier.model = svmtrain(trainLabels, trainData, options.training);
                classifier.posterior = @SVM;

            case 'SVMlight'
                classifier.model = svmlearn(trainData, trainLabels, options.training);
                classifier.posterior = @SVMlight;

            otherwise
                error('classifier not implemented');
        end
    end
end