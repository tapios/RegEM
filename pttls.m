function [Xr, Sr, rho, eta] = pttls(V, d, colA, colB, r)
%PTTLS Truncated TLS regularization with permuted columns.
%
%    Given matrices A and B, the total least squares (TLS) problem
%    consists of finding a matrix Xr that satisfies
%
%                (A+dA)*Xr = B+dB.
%
%    The solution must be such that the perturbation matrices dA
%    and dB have minimum Frobenius norm rho=norm( [dA dB], 'fro')
%    and each column of B+dB is in the range of A+dA [1].
%  
%    [Xr, Sr, rho, eta] = PTTLS(V, d, colA, colB, r) computes the
%    minimum-norm solution Xr of the TLS problem, truncated at rank r
%    [2]. The solution Xr of this truncated TLS problem is a
%    regularized error-in-variables estimate of regression
%    coefficients in the regression model A*X = B + noise(S). The
%    model may have multiple right-hand sides, represented as columns
%    of B.
%
%    As input, PTTLS requires the right singular matrix V of the
%    augmented data matrix C = U*diag(s)*V' and the vector d=s.^2 with
%    the squared singular values. Only right singular vectors V(:,j)
%    belonging to nonzero singular values s(j) are required.  Usually,
%    the first n columns of the augmented data matrix C correspond to
%    the n columns of the matrix A, and the k last columns to the k
%    right-hand sides B, so that the augmented data matrix is of the
%    form C=[A B]. PTTLS allows a more flexible composition of the
%    data matrix C: the columns with indices colA correspond to
%    columns of A; the columns with indices colB correspond to columns
%    of B.
%
%    The right singular vectors V and the squared singular values d
%    may be obtained from an eigendecomposition of the matrix C'*C = v
%    * d * v', which, for centered data, is proportional to the sample
%    covariance matrix.
%
%    PTTLS returns the rank-r truncated TLS solution Xr and the matrix
%    Sr = dB'*dB, which is proportional to the estimated covariance
%    matrix of the residual dB. Also returned are the Frobenius norm 
%
%                    rho = norm([dA dB], 'fro') 
%
%    of the residuals and the Frobenius norm 
%
%                    eta = norm(Xr,'fro') 
%
%    of the solution matrix Xr.
%  
%    If the truncation parameter r(1:nr) is a vector of length nr,
%    then Xr is a 3-D matrix with
%    
%         Xr(:,:, 1:nr) = [ Xr(r(1)), Xr(r(2)), ..., Xr(r(nr)) ] .
%
%    The covariance matrix estimate Sr has an analogous structure, and
%    the residual norm rho(1:nr) and the solution norm eta(1:nr) are
%    vectors with one element for each r(1:nr).
%
%    If r is not specified or if r > n, r = n is used.

%     References: 
%     [1] Van Huffel, S. and J. Vandewalle, 1991:
%         The Total Least Squares Problem: Computational Aspects
%         and Analysis. Frontiers in Mathematics Series, vol. 9. SIAM.
%     [2] Fierro, R. D., G. H. Golub, P. C. Hansen and D. P. O'Leary, 1997:
%         Regularization by truncated total least squares, SIAM
%         J. Sci. Comput., 18, 1223-1241
%     [3] Golub, G. H, and C. F. van Loan, 1989: Matrix
%         Computations, 2d ed., Johns Hopkins University Press,
%         chapter 12.3 

  %narginchk(2,5)          % check number of input arguments 

  [na,ma]   = size(V); 
  nd        = length(d);
  if nd ~= numel(d)
    error('Squared singular values d must be given as vector.')
  end
  if ma < nd
    error(['All right singular vectors with nonzero singular values' ...
	   ' are required.'])
  end
  d         = d(:);                   % make sure d is column vector
  
  if nargin == 2
    n       = na-1;                   % default number of columns of A
    k       = 1;                      % default number of right-hand sides
    colA    = 1:n;                    % take first n columns of [A B] as A 
    colB    = na;                     % take last column of [A B] as B
  else
    n       = length(colA);           % number of columns of A (number of variables)
    k       = length(colB);           % number of right-hand sides 
    if n + k ~= na
      error('Impossible set of column indices.')
    end
  end	     

  if (nargin < 5) 
    r = n;                            % default truncation of TLS 
  end
  
  ir = find(r > n);
  if ~isempty(ir)
    r(ir) = n;
    warning('Truncation parameter lowered.')
  end
      
  % initialize output variables
  nr = length(r);                     % number of truncation parameters
  Xr = zeros(n, k, nr);
  Sr = zeros(k, k, nr);
  if (nargout > 2)
    rho = zeros(nr,1);
  end
  if (nargout==4) 
    eta = zeros(nr,1); 
  end
  
  % compute a separate solution for each r
  for ir=1:nr
    rc           = r(ir);
    if rc > 0 % if rc == 0, Xr = 0
      V11        = V(colA, 1:rc);
      V21        = V(colB, 1:rc);
      Xr(:,:,ir) = (V11 / (V11'*V11)) * V21';
    
      % more traditional alternative formula when all right
      % singular vectors (or only those with small eigenvalues) are
      % given (NB: this requires all right singular vectors in V12
      % and V22, including those with *zero* singular values)
      % V12         = V(colA, rc+1:end);
      % V22         = V(colB, rc+1:end); 
      % Xr(:,:,ir)  = -V12/V22; 
    end
      
    % covariance matrix of residuals dB0'*dB0 (up to a
    % degrees-of-freedom factor)
    V22          = V(colB, rc+1:nd); 
    Sr(:,:,ir)   = V22 * (repmat(d(rc+1:nd), 1, k) .* V22');
      
    % covariance matrix of the total residuals R = A*Xr - B, which
    % is proportional (up to degree-of-freedom factor) to the
    % matrix St = R'*R (NB: V22 here again must contain all right
    % singular vectors, including those with zero singular values)
    % S2S2         = [diag(d(rc+1:nd)), zeros(nd-rc, size(V22, 2)-nd+rc); ...
    %                 zeros(size(V22, 2)-nd+rc, size(V22, 2))];
    % C            = V22' \ S2S2 / V22; 
    
    if (nargout > 2)       % total residual norm = norm( [dA0 dB0], 'fro')
      rho(ir)   = sqrt(sum(d(rc+1:nd)));     
    end
    
    if (nargout == 4)      % solution norm = norm( Xr, 'fro')
      eta(ir)   = norm(Xr(:,:,ir), 'fro'); 
    end
  end





