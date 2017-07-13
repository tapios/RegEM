function [in, out, nin] = kcvindices(n, K, n_add)
%KCVINDICES   Returns random indices for K-fold cross-validation
%
% [in, out, nin] = KCVINDICES(n, K) returns cell arrays in{1:K} and 
% out{1:K} containing indices to be included in a cross-validation sample 
% (in) and to be left out (out). 
%    
% The sample of size n (indices 1:n) is partitioned into K random 
% folds to be sequentially left out in K-fold cross validation. The
% folds to be left out for verification have approximately equal
% size. [Precisely, the first m=mod(n,K) folds have size l+1, the
% remaining folds size l, where l=floor(n/K).] Each CV sample 
% in{1:K} is of size nin{1:K}.
%
% [in, out, nin] = KCVINDICES(n, K, n_add) appends additional indices
% n+1 : n+n_add that are always to be included in the cross-validation 
% samples in{:}.

   %narginchk(2,3)          % check number of input arguments 
  
   if nargin < 3
       n_add = 0;
   end
   
   % get random permutation of indices 1:n (sample size n)
   P = randperm(n);
   
   % approximate length of folds
   l = floor(n/K);
   
   % modulus of division of n by K (number of folds that need l+1 items) 
   m = mod(n, K);
   
   % establish folds to be left out for verification: make the
   % first m folds 1 unit longer than subsequent folds to balance fold sizes
   out = cell(K,1);
   ks = 1;                           % start index in random permutation
   for k=1:K
       ke = (ks + l - 1) + (k <= m); % end index in random permutation
       out{k} = sort(P(ks:ke));
       ks = ke + 1;
   end
   
   % collect indices included in each KCV sample
   in  = cell(K, 1);
   nin = cell(K, 1);
   for k=1:K
      in{k}  = sort(cat(2, out{find(k ~= [1:K])}, n+1:n+n_add));
      nin{k} = length(in{k});
   end