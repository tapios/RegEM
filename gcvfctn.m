function g = gcvfctn(h, d, fc2, trS0, dof0)
%GCVFCTN    Evaluate object function for generalized cross-validation.
%
%   GCVFCTN(h, d, fc2, trS0, dof0) returns the function values of the
%   generalized cross-validation object function
%
%                     trace [ S0 + F' * diag(g.^2) * F ]
%              G(h) = ---------------------------------- 
%                           ( dof0 + sum(g) )^2
%
%   where g = h^2 ./ (d + h^2) = 1 - d.^2 ./ (d + h^2). The argument h
%   of the GCV function is the regularization parameter, and d is a
%   column vector of eigenvalues (see GCVRIDGE for the meaning of the
%   other symbols above). GCVFCTN is an auxiliary routine that is
%   called by GCVRIDGE. The input arguments are defined in GCVRIDGE:
%
%        h:  regularization parameter,
%        d:  column vector of eigenvalues of cov(X),
%      fc2:  row sum of squared Fourier coefficients, fc2=sum(F.^2, 2),
%     trS0:  trace(S0) = Frobenius norm of generic part of residual matrix,
%     dof0:  degrees of freedom in estimate of residual covariance
%            matrix when regularization parameter is set to zero
  
%   Adapted from GCVFUN in Per Christian Hansen's REGUTOOLS Toolbox.

  filfac = (h^2) ./ (d + h^2);            
  g      = ( sum(filfac.^2 .* fc2) + trS0 ) / (dof0 + sum(filfac))^2;
