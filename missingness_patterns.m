function [np, kavlr, kmisr, prows, mp, iptrn] = missingness_patterns(X)
%MISSINGNESS_PATTERNS  Returns unique patterns of missing values in a data matrix
%
%    [np, kavl, kmis, prows, mp, iptrn] = MISSINGNESS_PATTERNS(X) for a 
%    data matrix X with n rows and NaNs indicating missing values returns 
%   
%      np:          number of unique missingness patterns in data matrix X
%      kavl{1:np}:  indices of available values in each pattern
%      kmis{1:np}:  indices of missing values in each pattern
%      prows{1:np}: rows of data matrix belonging to each pattern
%      mp(1:np):    number of rows in pattern
%      iptrn(1:n):  index of pattern to which each row of data matrix
%                   belongs

  available    = ~isnan(X);
  [ptrns, ~, iptrn] = unique(available, 'rows'); 
  np           = size(ptrns, 1);                % number of unique patterns
  
  % For each missingness pattern, assemble the column indices of the available
  % values, of the missing values, and the data matrix rows corresponding
  % to the pattern
  kavlr        = cell(np,1);
  kmisr        = cell(np,1);
  prows        = cell(np,1);
  mp           = zeros(np, 1);
  for j=1:np
    kavlr{j}   = find(ptrns(j,:) == 1);
    kmisr{j}   = find(ptrns(j,:) == 0);
    prows{j}   = find(iptrn == j);              % rows of data matrix in pattern
    mp(j)      = length(prows{j});              % number of rows in pattern
  end