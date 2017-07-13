function h_opt = gcvridge(F, d, trS0, n, r, trSmin, options)
%GCVRIDGE   Finds minimum of GCV function for ridge regression.
%
%   GCVRIDGE(F, d, trS0, n, r, trSmin, OPTIONS) finds the
%   regularization parameter h that minimizes the generalized
%   cross-validation function
%
%                         trace S_h
%                 G(h) = ----------- 
%                          T(h)^2
%
%   of the linear regression model Y = X*B + E. The data matrices X
%   and Y are assumed to have n rows, and the matrix Y of dependent
%   variables can have multiple columns. The matrix S_h is the second
%   moment matrix S_h = E_h'*E_h/n of the residuals E_h = Y - X*B_h,
%   where B_h is, for a given regularization parameter h, the
%   regularized estimate of the regression coefficients,
% 
%                B_h = inv(X'*X + n h^2*I) * X'*Y.
%
%   The residual second second moment matrix S_h can be represented
%   as
%
%                S_h = S0 + F' * diag(g.^2) * F
%
%   where g = h^2 ./ (d + h^2) = 1 - d.^2 ./ (d + h^2) and d is a
%   column vector of eigenvalues of X'*X/n. The matrix F is the matrix
%   of Fourier coefficients. In terms of a singular value
%   decomposition of the rescaled data matrix n^(-1/2) * X = U *
%   diag(sqrt(d)) * V', the matrix of Fourier coefficients F can be
%   expressed as F = n^(-1/2) * U' * Y. In terms of the eigenvectors V
%   and eigenvalues d of X'*X/n, the Fourier coefficients are F =
%   diag(1./sqrt(d)) * V' * X' * Y/n. The matrix S0 is that part of
%   the residual second moment matrix that does not depend on the
%   regularization parameter: S0 = Y'*Y/n - F'*F.
% 
%   As input arguments, GCVRIDGE requires:
%        F:  the matrix of Fourier coefficients,
%        d:  column vector of eigenvalues of X'*X/n,
%     trS0:  trace(S0) = trace of generic part of residual 2nd moment matrix,
%        n:  number of degrees of freedom for estimation of 2nd moments,
%        r:  number of nonzero eigenvalues of X'*X/n,
%   trSmin:  minimum of trace(S_h) to construct approximate lower bound
%            on regularization parameter h (to prevent GCV from choosing
%            too small a regularization parameter).
%
%   The vector d of nonzero eigenvalues of X'*X/n is assumed to be
%   ordered such that the first r elements of d are nonzero and ordered 
%   from largest to smallest.
%  
%   The input structure OPTIONS contains optional parameters for the
%   algorithm:
%
%     Field name           Parameter                                  Default
%
%     OPTIONS.minvarfrac Minimum fraction of total variation in X     0
%                        that must be retained in the 
%                        regularization. From the parameter 
%                        OPTIONS.minvarfrac, an approximate upper 
%                        bound for the regularization parameter is
%                        constructed. The default value 
%                        OPTIONS.minvarfrac = 0  corresponds to no
%                        upper bound for the regularization parameter.   
  
%   References:
%   GCVRIDGE is adapted from GCV in Per Christian Hansen's REGUTOOLS
%       toolbox:
%   P.C. Hansen, "Regularization Tools: A Matlab package for
%       analysis and solution of discrete ill-posed problems,"
%       Numer. Algorithms, 6 (1994), 1--35.
%
%   see also: 
%   G. Wahba, "Spline Models for Observational Data",
%       CBMS_NSF Regional Conference Series in Applied Mathematics,
%       SIAM, 1990, chapter 4.

  %narginchk(6, 7)     % check number of input arguments 

  % sizes of input matrices
  d      = d(:);                   % make sure that d is column vector
  if length(d) < r
    error('All nonzero eigenvalues must be given.')
  end

  % ================           process options        ======================
  if nargin == 6 || isempty(options)
    fopts       = [];
  else
    fopts       = fieldnames(options);
  end
    
  minvarfrac    = 0;
  if strmatch('minvarfrac', fopts)
    minvarfrac = options.minvarfrac;
    if minvarfrac < 0 || minvarfrac > 1
      error('OPTIONS.minvarfrac must be in [0,1].')
    end
  end
  % ========================================================================
  
  p      = size(F, 1);
  if p < r
    error(['F must have at least as many rows as there are nonzero' ...
	   ' eigenvalues d.']) 
  end
  % row sum of squared Fourier coefficients
  fc2    = sum(F.^2, 2);
  
  % accuracy of regularization parameter 
  h_tol  = .2/sqrt(n);        
  
  % heuristic upper bound on regularization parameter
  varfrac = cumsum(d)/sum(d);
  if minvarfrac > min(varfrac)
    d_max           = interp1(varfrac, d, minvarfrac, 'linear');
    h_max           = sqrt( d_max );
  else            
    h_max           = sqrt( max(d) ) / h_tol;    
  end
    
  % heuristic lower bound on regularization parameter
  if trS0 > trSmin
    % squared residual norm is greater than a priori bound for all 
    % regularization parameters
    h_min         = sqrt(eps);
  else
    % find squared residual norms of truncated SVD solutions
    rtsvd         = zeros(r, 1);
    rtsvd(r)      = trS0;
    for j = r-1: -1: 1
      rtsvd(j)    = rtsvd(j+1) + fc2(j+1);
    end
    % take regularization parameter equal to square root of eigenvalue 
    % that corresponds to TSVD truncation level with residual norm closest 
    % to a priori bound trSmin
    [~, rmin] = min(abs(rtsvd - trSmin));
    h_min         = sqrt( max( d(rmin), min(d)/n ) );
  end

  if h_min < h_max
    % find minimizer of GCV function
    minopt = optimset('TolX', h_tol, 'Display', 'off');
    %h_opt  = fminbnd('gcvfctn', h_min, h_max, minopt, d(1:r), fc2(1:r), trS0, n-r);
    gcvfctn_to_minimize = @(h) gcvfctn(h, d(1:r), fc2(1:r), trS0, n-r);
    h_opt  = fminbnd(gcvfctn_to_minimize, h_min, h_max, minopt);
  else
    warning(['Upper bound on regularization parameter smaller' ...
	     ' than lower bound.'])
    h_opt  = h_min; 
  end

