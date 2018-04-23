
function scores = SVMlight(model, data, varargin)
    [error_rate, prediction] = svmclassify(data, zeros(size(data,1), 1), model);
end