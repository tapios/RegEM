function [B, S, h, peff] = iridge(Cxx, Cyy, Cxy, dof, options)
%IRIDGE  Individual ridge regressions with generalized cross-validation.
%
%   [B, S, h, peff] = IRIDGE(Cxx, Cyy, Cxy, dof) returns a regularized
%   estimate B of the coefficient matrix for the multivariate multiple
%   regression model Y = X*B + noise(S).  Each column B(:,k) of B is
%   computed by a ridge regression as B(:,k) = Mxx_hk Cxy(:,k), where
%   Mxx_hk is a regularized inverse of Cxx,
%
%             Mxx_h = inv(Cxx + hk^2 * I).
%
%   For each column k of B, an individual regularization parameter
%   ('ridge parameter') hk is selected as the minimizer of the
%   generalized cross-validation function. The matrix Cxx is an
%   estimate of the covariance matrix of the independent variables X,
%   Cyy is an estimate of the covariance matrix of the dependent
%   variables Y, and Cxy is an estimate of the cross-covariance matrix
%   of the independent variables X and the dependent variables Y. The
%   scalar dof is the number of degrees of freedom that were available
%   for the estimation of the covariance matrices.
%
%   The input structure OPTIONS contains optional parameters for the
%   algorithm:
%
%     Field name         Parameter                                   Default
%
%     OPTIONS.relvar_res Minimum relative variance of residuals.       5e-2
%                        From the parameter OPTIONS.relvar_res, a
%                        lower bound for the regularization parameter
%                        is constructed, in order to prevent GCV from
%                        erroneously choosing too small a 
%                        regularization parameter (see GCVRIDGE).
%
%   The OPTIONS structure is also passed to GCVRIDGE.
%     
%   IRIDGE returns an estimate B of the matrix of regression
%   coefficients. Also returned are an estimate S of the residual
%   covariance matrix, a vector h containing the regularization
%   parameters hk for the columns of B, and the scalar peff, an
%   estimate of the effective number of adjustable parameters in each
%   column of B.
% 
%   IRIDGE computes the estimates of the coefficient matrix and of the
%   residual covariance matrix from the covariance matrices Cxx, Cyy,
%   and Cxy by solving the regularized normal equations. The normal
%   equations are solved via an eigendecomposition of the covariance
%   matrix Cxx. However, if the data matrices X and Y are directly
%   available, a method based on a direct factorization of the data
%   matrices will usually be more efficient and more accurate.
%
%   See also: MRIDGE, GCVRIDGE.
  
  %narginchk(4, 5)     % check number of input arguments 
  
  px           = size(Cxx, 1);
  py           = size(Cyy, 1);
  if size(Cxx, 2) ~= px || size(Cyy, 2) ~= py || any(size(Cxy) ~= [px, py]) 
    error('Incompatible sizes of covariance matrices.')
  end

  % ==============           process options        ========================
  if nargin < 5 || isempty(options)
    options    = [];
    fopts      = [];
  else
    fopts      = fieldnames(options);
  end
    
  if strmatch('relvar_res', fopts)
    relvar_res = options.relvar_res; 
  else
    relvar_res = 5e-2;
  end
  
  % =================           end options        =========================

  if nargout > 1
    S_out      = 1==1;
  else
    S_out      = 0==1;
  end
  
  if nargout == 4
    peff_out   = 1==1;
  else
    peff_out   = 0==1;
  end
    
  % eigendecomposition of Cxx
  rmax         = min(dof, px);     % maximum possible rank of Cxx
  [V, d, r]    = peigs(Cxx, rmax);
  
  % Fourier coefficients. (The following expression for the Fourier
  % coefficients is only correct if Cxx = X'*X and Cxy = X'*Y for
  % some, possibly scaled and augmented, data matrices X and Y; for
  % general Cxx and Cxy, all eigenvectors V of Cxx must be included,
  % not just those belonging to nonzero eigenvalues.)
  F            = repmat(ones(r, 1)./sqrt(d), 1, px) .* V' * Cxy;

  % Part of residual covariance matrix that does not depend on the
  % regularization parameter h:
  if (dof > r) 
    S0         = Cyy - F'*F;
  else
    S0         = sparse(py, py);
  end
    
  % approximate minimum squared residual
  trSmin       = relvar_res * diag(Cyy);

  % initialize output
  h            = zeros(py, 1);
  B            = zeros(px, py);
  
  if S_out
    S          = zeros(py, py);
  end
  
  if peff_out
    peff       = zeros(py, 1);
  end
  
  for k = 1:py                    
    % compute an individual ridge regression for each y-variable
    
    % find regularization parameter that minimizes the GCV object function
    h(k)       = gcvridge(F(:,k), d, S0(k,k), dof, r, trSmin(k), options);
    
    % k-th column of matrix of regression coefficients
    B(:,k)     = V * (sqrt(d)./(d + h(k)^2) .* F(:,k));

    if S_out
      % assemble estimate of covariance matrix of residuals
      for j = 1:k
        diagS  = ( h(j)^2 * h(k)^2 ) ./ ( (d + h(j)^2) .* (d + h(k)^2) );
        S(j,k) = S0(j,k) + F(:,j)' * (diagS .* F(:,k));
        S(k,j) = S(j,k);
      end
    end

    if peff_out 
      % effective number of adjusted parameters in this column
      % of B: peff = trace(Mxx_h Cxx)
      peff(k)  = sum(d ./ (d + h(k)^2));
    end
    
  end

