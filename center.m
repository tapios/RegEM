function [xc, xm] = center(x)
%CENTER	  Centers data by subtraction of the mean.
%
%    [XC, XM] = CENTER(X) centers the data in X by subtraction of the
%    mean XM. If X contains NaNs, indicating missing values, the mean
%    of X is computed from the available data.
%
%    See also NANMEAN, MEAN.

  %narginchk(1,1)          % check number of input arguments 
  if ndims(x) > 2,  error('X must be vector or 2-D array.'); end
  
  % if x is a vector, make sure it is a row vector
  if length(x)==numel(x)         
    x   = x(:);                         
  end 
  m     = size(x, 1);

  % get mean of x
  if any(any(isnan(x)))               % there are missing values in x
    xm  = nanmean(x);
  else                                % no missing values
    xm  = mean(x);
  end

  % remove mean
  xc    = x - repmat(xm, m, 1);       
  
