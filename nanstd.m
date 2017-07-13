function sx = nanstd(x, flag)
%NANSTD   Standard deviation of available data, ignoring NaNs.
%
%    NANSTD(X) returns the standard deviation of the available data in
%    X, treating NaNs as missing values.  For vectors, NANSTD(X) is
%    the standard deviation of the non-NaN elements in X.  For
%    matrices, NANSTD(X) is a row vector containing the standard
%    deviation of the non-NaN elements in each column.
%
%    NANSTD(X) normalizes by (N-1) where, for each element of
%    NANSTD(X), N is number of available values.
%
%    NANSTD(X,0) normalizes by N and produces the second moment of the
%    available data about their mean.  NANSTD(X,1) is the same as
%    NANSTD(X).
%  
%    See also STD, NANMEAN.

  % maximum admissible fraction of missing values
  max_miss = 0.6;                

  %narginchk(1,2)          % check number of input arguments 
  
  if isempty(x)                       % check for empty input.
    sx = NaN;
    return
  end
  if ndims(x) > 2,  error('Data must be vector or 2-D array.'); end

  if nargin < 2, flag = 1; end        % default: normalize by nobs-1

  % if x is a vector, make sure it is a row vector
  if length(x)==numel(x)         
    x = x(:);                         
  end  
  [m,n]   = size(x);
    
  % determine number of available observations on each variable
  inan    = find(isnan(x));
  [i,j]   = ind2sub([m,n], inan);     % subscripts of missing entries
  nans    = sparse(i,j,1,m,n);        % indicator matrix for missing values
  nobs    = m - sum(nans);
  
  % set nobs to NaN when there are too few entries to form robust average
  minobs  = m * (1 - max_miss);
  k       = find(nobs < minobs);
  nobs(k) = NaN;
  
  % center data
  xc      = x - repmat(nanmean(x), m, 1);
  
  % replace NaNs with zeros in centered data matrix
  xc(inan) = zeros(size(inan));
  
  % standard deviation
  sx      = sqrt(sum(conj(xc).*xc) ./ (nobs-flag));

