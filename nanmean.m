function mx = nanmean(x)
%NANMEAN  Mean of available data, ignoring NaNs.
%
%    NANMEAN(X) returns the mean of the available data in X, treating
%    NaNs as missing values. For vectors, NANMEAN(X) is the mean value
%    of the non-NaN elements in X.  For matrices, NANMEAN(X) is a row
%    vector containing the mean value of each column, ignoring NaNs.
%
%    If, in forming the mean, the fraction of missing terms exceeds
%    a critical value, the mean is set to NaN.
%  
%    See also MEAN, NANSTD, NANSUM.

  % maximum admissible fraction of missing values
  max_miss = 0.99;                
  
  %narginchk(1,1)          % check number of input arguments 
  if isempty(x)                       % check for empty input.
    mx = NaN;
    return
  end

  % if x is vector, make sure it is a row vector
  if length(x)==numel(x)         
    x = x(:);                         
  end
  [m,n]   = size(x);
  
  % replace NaNs with zeros.
  inan    = find(isnan(x));
  x(inan) = zeros(size(inan));
  
  % determine number of available observations on each variable
  [i,j]   = ind2sub([m,n], inan);     % subscripts of missing entries
  nans    = sparse(i,j,1,m,n);        % indicator matrix for missing values
  nobs    = m - sum(nans);
    
  % set nobs to NaN when there are too few entries to form robust average
  minobs  = m * (1 - max_miss);
  k       = find(nobs < minobs);
  nobs(k) = NaN;
  
  mx      = sum(x) ./ nobs;


