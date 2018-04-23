
function scores = SVM(model, data, varargin)
    options = varargin{1};
    [prediction, accuracy, scores] = svmpredict(zeros(size(data,1), 1), data, model, options.testing);
end