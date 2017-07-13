function [X, C, D] = rescale(X, C, D)
% RESCALE     Rescales variables and covariance matrix to unit variance.
%
% [Xs, Cs, D] = RESCALE(X, C) rescales the data matrix X and the
% covariance matrix C such that the scaled data matrix Xs and
% covariance matrix Cs have unit variance. Constant variables (with
% zero variance) are not rescaled.
%
% [Xs, Cs] = RESCALE(X, C, D) rescales the data matrix X and the
% covariance matrix C with the scale factors in vector D (by
% dividing variables by the elements of D).
%
% See also: STANDARDIZE, CENTER

    [n, p]     = size(X);
    if nargin < 3
        D      = sqrt(diag(C)); 
    end
    const      = (abs(D) < eps);   % test for constant variables
    nconst     = ~const;
    if sum(const) ~= 0             % do not scale constant variables
      D        = D .* nconst + 1*const;
    end
    X          = X ./ repmat(D', n, 1);
    % correlation matrix
    C          = C ./ repmat(D', p, 1) ./ repmat(D, 1, p);
