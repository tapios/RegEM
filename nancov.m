function [c, xm, xc] = nancov(x, flag)
%NANCOV  Sample covariance matrix of available data, ignoring NaNs.
% 
%    NANCOV(X) returns the covariance matrix of X estimated from all
%    available data in X, ignoring NaN-elements that indicate missing
%    values. If X is a vector, NANCOV(X) returns the variance. For
%    matrices, where each row is an observation and each column a
%    variable, NANCOV(X) is the covariance matrix estimate from all
%    available data.
%
%    NANCOV(X) normalizes each covariance matrix element by (Nt-1)
%    where Nt is the number of available product terms contributing to
%    the covariance matrix element.
%
%    NANCOV(X,0) normalizes by Nt and produces the second moment
%    matrix of the observations about their mean.  NANCOV(X,1) is the
%    same as NANCOV(X).
%  
%    If all terms in the sum of the product and/or cross-product terms
%    contributing to a covariance matrix element are missing, the
%    covariance matrix element is set to zero.
%
%    Because the covariance matrix elements are normalized with
%    different normalization factors, depending on how many terms
%    contribute to them, the covariance matrix is not necessarily
%    positive semidefinite.
%
%    [C, M, XC] = NANCOV(X) returns the covariance matrix estimate
%    C from all available data, the mean M of the available data,
%    and the centered data matrix XC with the mean removed from
%    each column. 
%
%    See also COV, CENTER, NANSTD, NANMEAN, NANSUM.
   
  %narginchk(1,2)          % check number of input arguments 
  if ndims(x) > 2,  error('Data must be vector or 2-D array.'); end
  
  if nargin < 2, flag = 1; end        % default: normalize by nt-1

  if length(x)==numel(x)         % if x is vector, make sure
    x = x(:);                         % it is a row vector
  end 
  
  % center data
  [xc,xm] = center(x);
  [m,n]   = size(xc);
  
  % replace NaNs with zeros.
  inan    = find(isnan(xc));
  xc(inan)= zeros(size(inan));
    
  nonnan  = ones(m,n);                % indicator matrix showing where there
  nonnan(inan) = zeros(size(inan));   % are non-missing  elements in xc
  nt      = nonnan' * nonnan - flag;  % normalization factor for each 
				      % covariance matrix element
    
  % set covariance to 0 when there are no terms to form sum
  k       = find(nt < 1);
  nt(k)   = 1;

  c       = xc' * xc ./ nt;



