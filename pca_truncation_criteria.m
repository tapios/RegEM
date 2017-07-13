function [mdl, ne08, aic, aicc] = pca_truncation_criteria(d, p, r, N, b)
%PCA_TRUNCATION_CRITERIA Computes criteria for truncating 
%   principal component analyses. 
%
% [MDL, NE08, AIC, AICC] = PCA_TUNCATION_CRITERIA(d, p, r, N, b) returns the
% truncation choice criteria Mean Description Length, Akaike's Information
% Criterion, and the bias-corrected AIC, given as inputs a vector of 
% eigenvalues d, the problem dimension p, possible truncation levels r, and 
% sample size N. The optional argument b indicates whether the problem is
% real-valued (b=1) or complex-valued (b=2); the default is b=1.
%
% The implementation of the truncation choice criteria follows Wax
% & Kailath, "Detection of Signals by Information Theoretic
% Criteria," IEEE Trans. Acoust. Speech Signal Proc., 33 (1985),
% 387--392. It uses Wax & Kailath's log-likelihood function and
% with that computes various information-theoretic criteria to
% select the truncation level:
%  
%    MDL: Schwartz (1978) and Rissanen (1978)
%    NE08: Nadakuditi & Edelman (2008) [for r<<p]
%    AIC: Akaike (1973, 1974) [times 1/2 to obtain likelihood measure]
%    AICC: Hurvich & Tsai (1989) [times 1/2]
%
% If the problem is rank-deficient (number of nonzero eigenvalues d < p),
% it uses an ad hoc restricted log-likelihood. In those cases, NE08 may be
% the preferred criterion (as it is designed to handle such cases).

% Tapio Schneider, 2/18/2012

if nargin < 5
  % default: real-valued problem
  b = 1;     
end

% sort eigenvalues (in case they are not)
d = sort(d(:), 1, 'descend');

% number of numerically nonzero eigenvalues
d_min = eps * max(d) * p;
nd = sum(d > d_min);

rmax = length(d) - 1;
if max(r) > rmax
  irb = find(r <= rmax);
  r   = r(irb);
  warning('Maximum truncation level was too large and was lowered.')
end

% Assemble log-likelihood (Wax & Kailath 1985) and 
% Nadakuditi & Edelman (2008) statistic for various truncations r
llh  = zeros(size(r));
tr   = zeros(size(r));
for j=1:length(r)
  k = r(j);
  % Wax & Kailath log-likelihood (ad hoc restricted to nonsingular subspace 
  % if nd<p, i.e., if sample covariance matrix is singular)
  llh(j) = N*(nd - k) ...
        * log( prod(d(k+1:nd))^(1/(p-k)) / sum(d(k+1:nd))*(p-k) );

  % Nadakuditi & Edelman statistic  
  tr(j) = p * ((p-k) * sum(d(k+1:nd).^2)/sum(d(k+1:nd))^2 - (1+p/N))...
      - (2/b-1)*p/N;
end

% Number of free parameters on which log-likelihood depends (number of
% eigenvector and eigenvalue components, minus the number of
% constraints for orthogonality and unit norm of eigenvectors)
%
% This number is obtained as follows:
%
% We have k + 1 + b*k*p parameters for eigenvalues, eigenvectors,
% and noise variance. From that subtract the number of constraints:
% -- normalization: b[(k-1) + (k-1) + ... + 1] = bk/2(k-1) constraints
% -- orthogonality: b*k constraints
% Total: k*(bp - (b/2)k - b/2 + 1) + 1 free parameters
peff = r.*(b*p - b/2*r - b/2 + 1) + 1;

% Assemble truncation choice criteria
mdl  = -llh + peff*log(N)/2;
ne08 = b/4*(N/p)^2*tr.^2 + 2*(r + 1);

% For AIC(c), use 1/2 the traditional measure (to obtain a likelihood measure)
aic  = -llh + peff;  
aicc = aic + peff.*(peff+1)./(N-peff-1);

end

