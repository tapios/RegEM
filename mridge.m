function [B, S, h, peff] = mridge(Cxx, Cyy, Cxy, dof, options)
%MRIDGE  Multiple ridge regression with generalized cross-validation.
%
%   [B, S, h, peff] = MRIDGE(Cxx, Cyy, Cxy, dof, OPTIONS) returns a
%   regularized estimate B = Mxx_h Cxy of the coefficient matrix in
%   the multivariate multiple regression model Y = X*B + noise(S). The
%   matrix Mxx_h is the regularized inverse of the covariance matrix
%   Cxx,
%
%             Mxx_h = inv(Cxx + h^2 * I).
%
%   The matrix Cxx is an estimate of the covariance matrix of the
%   independent variables X, Cyy is an estimate of the covariance
%   matrix of the dependent variables Y, and Cxy is an estimate of the
%   cross-covariance matrix of the independent variables X and the
%   dependent variables Y. The scalar dof is the number of degrees of
%   freedom that were available for the estimation of the covariance
%   matrices.
%
%   The input structure OPTIONS contains optional parameters for the
%   algorithm:
%
%     Field name         Parameter                                   Default
%
%     OPTIONS.regpar     Regularization parameter h. If regpar       not set
%                        is set, the scalar OPTIONS.regpar is
%                        taken as the regularization parameter h. 
%                        If OPTIONS.regpar is not set (default), 
%                        the regularization parameter h is selected 
%                        as the minimizer of the generalized 
%                        cross-validation (GCV) function. The output
%                        variable h then contains the selected 
%                        regularization parameter.
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
%   MRIDGE returns the ridge estimate B of the matrix of regression
%   coefficients. Also returned are an estimate S of the residual
%   covariance matrix, the regularization parameter h, and the scalar
%   peff, an estimate of the effective number of adjustable
%   parameters in B.  
%  
%   MRIDGE computes the estimates of the coefficient matrix and of the
%   residual covariance matrix from the covariance matrices Cxx, Cyy,
%   and Cxy by solving the regularized normal equations. The normal
%   equations are solved via an eigendecomposition of the covariance
%   matrix Cxx. However, if the data matrices X and Y are directly
%   available, a method based on a direct factorization of the data
%   matrices will usually be more efficient and more accurate.
%
%   See also: IRIDGE, GCVRIDGE.

  %narginchk(4, 5)     % check number of input arguments 
  
  px           = size(Cxx, 1);
  py           = size(Cyy, 1);
  if size(Cxx, 2) ~= px || size(Cyy, 2) ~= py || any(size(Cxy) ~= [px, py]) 
    error('Incompatible sizes of covariance matrices.')
  end
  
  % ==============           process options        ========================
  if nargin == 4 || isempty(options)
    options    = [];
    fopts      = [];
  else
    fopts      = fieldnames(options);
  end
    
  if strmatch('regpar', fopts)
    h          = options.regpar; 
    h_given    = 1;
  else
    h_given    = 0;
  end
  
  if strmatch('relvar_res', fopts)
    relvar_res = options.relvar_res; 
  else
    relvar_res = 5e-2;
  end
  % =================           end options        =========================
  
  if nargout > 1
    S_out      = 1;
  else
    S_out      = 0;
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
  
  if ~h_given
    % approximate minimum squared residual
    trSmin     = relvar_res * trace(Cyy);
    
    % find regularization parameter that minimizes the GCV object function
    h          = gcvridge(F, d, trace(S0), dof, r, trSmin, options);
  end
 
  % get matrix of regression coefficients
  B            = V * (repmat(sqrt(d)./(d + h^2), 1, py) .* F);
  
  if S_out
    % get estimate of covariance matrix of residuals
    S          = S0 + F' * (repmat(h^4./(d + h^2).^2, 1, py) .* F);
  end
  
  if nargout == 4
    % effective number of adjusted parameters: peff = trace(Mxx_h Cxx)
    peff       = sum(d ./ (d + h^2));
  end      


