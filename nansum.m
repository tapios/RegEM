function y = nansum(x)
%NANSUM   Sum ignoring NaNs.
%
%    NANSUM(X) returns the sum over non-NaN elements of X.  For
%    vectors, NANSUM(X) is the sum of the non-NaN elements in X. For
%    matrices, NANSUM(X) is a row vector containing the sum of the
%    non-NaN elements in each column of X.
%
%    See also NANMEAN, NANSTD.

  %narginchk(1,1)          % check number of input arguments 
				      
  % replace NaNs with zeros.
  nans    = isnan(x);
  inan    = find(nans);
  x(inan) = zeros(size(inan));

  y       = sum(x);
  
  % protect against an entire column of NaNs
  iall    = find(all(nans));
  y(iall) = NaN;

