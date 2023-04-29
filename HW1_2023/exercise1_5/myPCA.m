function [U, S] = myPCA(X)
%   PCA Run principal component analysis on the dataset X
%   [U, S, X] = myPCA(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE GOES HERE ======================
% Apply PCA by computing the eigenvectors and eigenvalues of the covariance matrix. 
%

C = (1/m)*transpose(X)*X;  % Covariance matrix (Identical to the autocorrelation because E[x]=0)
[U, S] = eig(C);
%U= -U; % To match comments' instructions 

% =========================================================================

end
