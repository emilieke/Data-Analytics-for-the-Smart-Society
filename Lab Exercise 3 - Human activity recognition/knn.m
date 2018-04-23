function scores = knn(model, data, varargin)
    options = varargin{1};
    [k] = process_options(options.testing, 'k', 5);

    [IDX, distances] = knnsearch(model.traindata, data, 'k', k);
    nLabels = length(unique(model.labels'));
    scores(1:length(IDX), 1:nLabels) = 0; % init to speed up
    for point = 1:size(IDX, 1)
        kn_labels = model.labels(IDX(point, :));

        for iLabel = unique(kn_labels)'
            scores(point, iLabel) = size(find(kn_labels==iLabel), 1) / k;
        end
    end
end