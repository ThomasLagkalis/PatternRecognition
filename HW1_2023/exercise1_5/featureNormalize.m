function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

[nSamples, nFeatures] = size(X);

for i=1:nFeatures
    mu= mean(X(:, i)); % mean of each column (feature)
    sigma= std(X(:, i)); % standart deviation of each column
    X_norm(:,i) = (X(:,i) - mu)/sigma; % normalize each column independently
end

mu=mean(X_norm); sigma=std(X_norm);

% ============================================================

end
