% -------------------------------------------------------------------------------
% Classification
% -------------------------------------------------------------------------------

function [scores] = classification(model, data, varargin)

    if iscell(model)
        assert(iscell(data)); % support same method type on its own data
        scores = cell(length(model), 1);

        for c = 1:length(model)
            d = data{c}(:);
            scores{c} = model{c}.posterior(model{c}.model, d, options);
        end   
    else
        d = data(:);
        scores = model.posterior(model.model, d, options);
    end
end

