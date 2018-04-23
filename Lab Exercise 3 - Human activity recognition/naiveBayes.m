% Naive Bayes
function scores = naiveBayes(model, data, varargin)
    scores = model.posterior(data);
end
