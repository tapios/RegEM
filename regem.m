function [X, M, C, Xerr, B, peff, kavlr, kmisr, iptrn] = regem(X, options)
%REGEM   Imputation of missing values with regularized EM algorithm.
%
%    [X, M, C, Xerr] = REGEM(X, OPTIONS) replaces missing values
%    (NaNs) in the data matrix X with imputed values. REGEM
%    returns
%  
%       X:     data matrix with imputed values substituted for NaNs,  
%       M:     estimated mean of X, 
%       C:     estimated covariance matrix of X,
%       Xerr:  estimated mean error of the imputed values.
%  
%    Missing values are imputed with a regularized expectation
%    maximization (EM) algorithm. In an iteration of the EM algorithm,
%    given estimates of the mean and of the covariance matrix are
%    revised in three steps. First, for each record X(i,:) (row of X, i=1:n) 
%    with missing values, the regression parameters of the variables with
%    missing values on the variables with available values are
%    computed from the estimates of the mean and of the covariance
%    matrix. Second, the missing values in a record X(i,:) are filled
%    in with their conditional expectation values given the available
%    values and the estimates of the mean and of the covariance
%    matrix, the conditional expectation values being the product of
%    the available values and the estimated regression
%    coefficients. Third, the mean and the covariance matrix are
%    re-estimated, the mean as the sample mean of the completed
%    dataset and the covariance matrix as the sum of the sample
%    covariance matrix of the completed dataset and an estimate of the
%    conditional covariance matrix of the imputation error. As the 
%    regression models need to be computed only for each unique pattern 
%    of missing values in records (rows of X), the iterations are go over 
%    unique missingness patterns. (This is considerably faster than 
%    iterating over each row of X, e.g., when missingness has a staircase 
%    pattern with few steps.)
%
%    In the regularized EM algorithm, the parameters of the regression
%    models are estimated by a regularized regression method. By
%    default, the parameters of the regression models are estimated by
%    a multiple ridge regression for each record, with one regularization 
%    parameter (ridge parameter) per record.  Optionally, the parameters 
%    of the regression models can be estimated by an individual ridge 
%    regression for each missing value, with one regularization parameter
%    per missing value. The regularization parameters for the ridge
%    regressions are selected as the minimizers of the generalized 
%    cross-validation (GCV) function, unless they are fixed and a priori 
%    specified. As another option, the parameters of the regression models 
%    can be estimated by truncated total least squares (TTLS). The truncation 
%    parameter, a discrete regularization parameter, can be either fixed 
%    and given or can be chosen in each iteration by one of a variety of 
%    selection criteria, from simple (and fast) information-theoretic 
%    criteria to more reliable (but slower) K-fold cross-validation. 
% 
%    As default initial condition for the imputation algorithm, the
%    mean of the data is computed from the available values, mean
%    values are filled in for missing values, and a covariance matrix
%    is estimated as the sample covariance matrix of the completed
%    dataset with mean values substituted for missing
%    values. Optionally, initial estimates for the missing values and
%    for the covariance matrix estimate can be given as input
%    arguments.   
%
%    To obtain good initial values for regularized EM iterations, the 
%    regularized EM algorithm may be used with TTLS and a fixed (relatively 
%    small) truncation parameter. This is very fast, as it requires only 
%    one eigendecomposition per iteration (instead of one eigendecomposition 
%    per missingness pattern and iteration with ridge regression, 
%    or K decompositions with K-fold cross validation).   
%
%    [X, M, C, Xerr, B, peff, kavlr, kmisr, iptrn] = REGEM(X, OPTIONS)
%    additionally returns the cell arrays B, peff, kavlr, kmisr (length equal to 
%    number of unique missingness patterns) and the vector iptrn (length n), 
%    which contain
%
%       B{j}:     Matrix of regression coefficents of missing values on
%                 available values for each missingness pattern. (This is for
%                 data centered with the estimate of the mean at the beginning
%                 of the last iteration, which differs from the mean vector 
%                 that is output by the update after the last iteration.)
%       peff{j}:  Effective number of degrees of freedom for each
%                 missingness pattern (and each missing value in pattern)
%       kavlr{j}: Indices of available values for each missingness pattern
%       kmisr{j}: Indices of missing values for each missingness pattern
%       iptrn(i): Index j of missingness pattern for row i of X 
% 
%    The OPTIONS structure specifies parameters in the algorithm:
%
%     Field name         Parameter                                  Default
%
%     OPTIONS.regress    Regression procedure to be used:           'mridge'
%                        'mridge': multiple ridge regression
%                        'iridge': individual ridge regressions
%                        'ttls':   truncated total least squares 
%                                  regression 
%  
%     OPTIONS.stagtol    Stagnation tolerance: quit when            1e-2 
%                        consecutive iterates of the missing
%                        values are so close that
%                          norm( Xmis(it)-Xmis(it-1) ) 
%                             <= stagtol * norm( Xmis(it-1) )
%  
%     OPTIONS.maxit      Maximum number of EM iterations.           50
%  
%     OPTIONS.inflation  Inflation factor for the residual          1 
%                        covariance matrix. Because of the 
%                        regularization, the residual covariance 
%                        matrix underestimates the conditional 
%                        covariance matrix of the imputation 
%                        error. The inflation factor is to correct 
%                        this underestimation. The update of the 
%                        covariance matrix estimate is computed 
%                        with residual covariance matrices 
%                        inflated by the factor OPTIONS.inflation,
%                        and the estimates of the imputation error
%                        are inflated by the same factor. 
%
%     OPTIONS.truncslct  Truncation parameter selection with TTLS   KCV
%                        -- Global methods: 'MDL', 'AIC', 
%                        'AICC', and 'NE08' choose a global 
%                        truncation parameter in each iteration 
%                        according to an information criterion for 
%                        truncating principal component analyses 
%                        (for details, see PCA_TRUNCATION_CRITERIA).
%                        -- Local method: 'KCV' chooses a 
%                        truncation adaptively for each record
%                        by K-fold cross-validation.
%
%     OPTIONS.Kcv        Parameter K for K-fold cross-validation    5 
%                        (only used if OPTIONS.truncslct='KCV')
%
%     OPTIONS.cv_norm    Norm of error to be minimized in K-fold    2
%                        cross-validation (only used if 
%                        OPTIONS.truncslct='KCV'). The mean Xerr 
%                        if KCV is chosen is also given in the 
%                        cv_norm.
%                
%                        (only used if OPTIONS.truncslct='KCV')
%
%     OPTIONS.ncv        Only rows 1:ncv are sequentially left      n
%                        out in cross validation (if 
%                        OPTIONS.truncslct='KCV'). Restricting 
%                        ncv to less than the sample size n may 
%                        be helpful if missing values only or 
%                        primarily occur in rows ncv+1:n of X.    
%
%     OPTIONS.regpar     Regularization parameter.                  not set 
%                        -- For ridge regression, set regpar to 
%                        sqrt(eps) for mild regularization; leave 
%                        regpar unset for GCV selection of
%                        regularization parameters.
%                        -- For TTLS regression, set regpar to a 
%                        number for a fixed truncation parameter. 
%                        Leave regpar unset to choose the 
%                        truncation parameter adaptively by 
%                        the method specified in OPTIONS.truncslct.
%                        Or specify a vector regpar of possible 
%                        truncation parameters from which an
%                        an optimal parameter is then chosen 
%                        adaptively by OPTIONS.truncslct. 
%    
%     OPTIONS.scalefac   Vector of scale factors with which         not set
%                        variables are to be rescaled prior to 
%                        regression analysis and regularization.
%                        (Variables will be divided by the scale
%                        factors). If OPTIONS.scalefac is not set,
%                        the estimated standard deviations will  
%                        be used as scale factors (so correlation
%                        matrices will be regularized).
%    
%     OPTIONS.relvar_res Minimum relative variance of residuals.    5e-2
%                        From the parameter OPTIONS.relvar_res, a
%                        lower bound for the regularization 
%                        parameter is constructed if GCV is used, 
%                        to prevent GCV from erroneously choosing 
%                        too small a regularization parameter.
%  
%     OPTIONS.minvarfrac Minimum fraction of total variation in     0
%                        standardized variables that must be 
%                        retained in the regularization.
%                        From the parameter minvarfrac, 
%                        an approximate upper bound for the 
%                        regularization parameter is constructed 
%                        if GCV is used. The default value 
%                        OPTIONS.minvarfrac = 0 corresponds to 
%                        no upper bound for the regularization 
%                        parameter.   
%  
%     OPTIONS.Xmis0      Initial imputed values. Xmis0 is a         not set
%                        (possibly sparse) matrix of the same 
%                        size as X with initial guesses in place
%                        of the NaNs in X.  
%  
%     OPTIONS.C0         Initial estimate of covariance matrix.     not set
%                        If no initial covariance matrix C0 is 
%                        given but initial estimates Xmis0 of the 
%                        missing values are given, the sample 
%                        covariance matrix of the dataset 
%                        completed with initial imputed values is 
%                        taken as an initial estimate of the 
%                        covariance matrix. 
%  
%     OPTIONS.Xcmp       Display the weighted rms difference        not set
%                        between the imputed values and the 
%                        values given in Xcmp, a matrix of the 
%                        same size as X but without missing 
%                        values. By default, REGEM displays 
%                        the rms difference between the imputed 
%                        values at consecutive iterations. The 
%                        option of displaying the difference 
%                        between the imputed values and reference 
%                        values exists for testing purposes.
%
%     OPTIONS.neigs      Number of eigenvalue-eigenvector pairs     not set
%                        to be computed for TTLS regression. 
%                        By default, all nonzero eigenvalues and 
%                        corresponding eigenvectors are computed. 
%                        By computing fewer (neigs) eigenvectors, 
%                        the computations can be accelerated, but 
%                        the residual covariance matrices become 
%                        inaccurate. The residual covariance 
%                        matrices underestimate the imputation 
%                        error conditional covariance matrices 
%                        more and more as neigs is decreased.
  
%    References: 
%    [1] T. Schneider, 2001: Analysis of incomplete climate data:
%        Estimation of mean values and covariance matrices and
%        imputation of missing values. Journal of Climate, 14,
%        853--871.  
%    [2] R. J. A. Little and D. B. Rubin, 1987: Statistical
%        Analysis with Missing Data. Wiley Series in Probability
%        and Mathematical Statistics. (For EM algorithm.) 
%    [3] P. C. Hansen, 1997: Rank-Deficient and Discrete Ill-Posed
%        Problems: Numerical Aspects of Linear Inversion. SIAM
%        Monographs on Mathematical Modeling and Computation.
%        (For regularization techniques, including the selection of 
%        regularization parameters.)
  
  %narginchk(1, 2)     % check number of input arguments 
  
  if ndims(X) > 2,  error('X must be vector or 2-D array.'); end
  % if X is a vector, make sure it is a column vector (representing 
  % a single variable)
  if length(X)==numel(X)      
    X = X(:);                      
  end 
  [n, p]       = size(X);
  
  % algorithm exploits covariance among variables; it will not work
  % with just one variable
  if p < 2,  error('X must contain at least 2 variables.'); end
  
  % number of degrees of freedom for estimation of covariance matrix
  dofC         = n - 1;            % use degrees of freedom correction       
  
  % ==============           process options        =======================
  if nargin ==1 || isempty(options)
    fopts      = [];
  else
    fopts      = fieldnames(options);
  end

  % initialize options structure for regression modules
  optreg       = struct;
  
  % default values
  regpar_given = 0==1;
  truncslct    = 'none';
  trial_trunc_given = 0==1;
 
  if ~isempty(strmatch('regress', fopts))
    regress    = lower(options.regress);
    switch regress
     case {'mridge', 'iridge'}
      if strmatch('regpar', fopts)
        regpar_given = 1;
        if ~isscalar(options.regpar)
          error('Regularization parameter must be a number')
        else
          optreg.regpar = options.regpar;
          regress       = 'mridge';
        end
      end
  
     case {'ttls'}
      
      if strmatch('truncslct', fopts)
          truncslct     = lower(options.truncslct);
      else  % default: K-fold CV
          truncslct     = 'kcv'; 
      end
      
      if ~isempty(strmatch('regpar', fopts)) && isscalar(options.regpar)
          % fixed truncation parameter given
          truncslct     = 'none';
          regpar_given  = 1==1;
          trunc         = min([options.regpar, n-1, p]);
      elseif ~isempty(strmatch('regpar', fopts)) && isvector(options.regpar)
          % vector of trial truncations given
          trial_trunc   = sort(options.regpar);
          trial_trunc_given = 1==1;
      else
          % default: try all possible truncations
          trial_trunc   = [];  
          trial_trunc_given = 0==1;
      end
      
      % number of cross-validation folds
      if isempty(strmatch('Kcv', fopts))
        Kcv    = 5;
      else
        Kcv    = options.Kcv; 
      end
      
      % norm of error to be minimized by K-fold cross-validation
      if isempty(strmatch('cv_norm', fopts))
        cv_norm    = 2;
      else
        cv_norm    = options.cv_norm; 
      end
         
      % maximum index to be included in cross-validation
      if isempty(strmatch('ncv', fopts))
        ncv    = n;       % default: cross-validation over entire sample
      else
        ncv    = options.ncv; 
      end
      
      if strmatch('neigs', fopts)
        neigs  = options.neigs;
      else
        neigs  = min(n-1, p);
      end
      
     otherwise
      
      error(['Unknown regression method ', regress])
      
    end
    
  else
    
      regress    = 'mridge';
  
  end
  
  if strmatch('stagtol', fopts)
    stagtol      = options.stagtol;
  else
    stagtol      = 1e-2;
  end

  if strmatch('maxit', fopts)
    maxit        = options.maxit;
  else
    maxit        = 30;
  end
  
  if strmatch('inflation', fopts)
    inflation    = options.inflation;
  else
    inflation    = 1;
  end
  
  if strmatch('scalefac', fopts)
    D_given      = 1;
    D            = options.scalefac(:); 
    if length(D) ~= p
        error('OPTIONS.scalefac must be vector of length p.')
    end
  else
    D_given      = 0;
  end

  
  if strmatch('relvar_res', fopts)
    optreg.relvar_res = options.relvar_res; 
  else
    optreg.relvar_res = 5e-2; 
  end
  
  if strmatch('minvarfrac', fopts)
    optreg.minvarfrac = options.minvarfrac; 
  else
    optreg.minvarfrac = 0; 
  end
    
  if strmatch('Xmis0', fopts);
    Xmis0_given= 1==1;
    Xmis0      = options.Xmis0;
    if any(size(Xmis0) ~= [n,p])
      error('OPTIONS.Xmis0 must have the same size as X.')
    end
  else
    Xmis0_given= 0==1;
  end
  
  if strmatch('C0', fopts);
    C0_given   = 1==1;
    C0         = options.C0;
    if any(size(C0) ~= [p, p])
      error('OPTIONS.C0 has size incompatible with X.')
    end
  else
    C0_given   = 0==1;
  end
  
  if strmatch('Xcmp', fopts);
    Xcmp_given = 1==1;
    Xcmp       = options.Xcmp;
    if any(size(Xcmp) ~= [n,p])
      error('OPTIONS.Xcmp must have the same size as X.')
    end
    sXcmp      = std(Xcmp);
  else
    Xcmp_given = 0==1;
  end
  
  % =================           end options        ========================
  
  % get indices of missing values and initialize matrix of imputed values
  indmis       = find(isnan(X));
  nmis         = length(indmis);
  if nmis == 0
    warning('No missing value flags found.')
    return                                      % no missing values
  end
  [~,kmis]  = ind2sub([n, p], indmis);
  % use full matrices: needs more memory but is faster
  Xmis          = X;                            % matrix of imputed values
  Xerr          = zeros(size(X));               % standard error imputed vals.
  
  % alternative: sparse matrices to save some memory
  %[jmis,kmis]  = ind2sub([n, p], indmis);
  %Xmis         = sparse(jmis, kmis, NaN, n, p); % matrix of imputed values
  %Xerr         = sparse(jmis, kmis, Inf, n, p); % standard error imputed vals.
  
  % find unique missingness patterns
  [np, kavlr, kmisr, prows, mp, iptrn] = missingness_patterns(X);
  
  disp(sprintf('\nREGEM:'))
  disp(sprintf('\tSample size:                     %4i', n))
  disp(sprintf('\tUnique missingness patterns:     %4i', np))
  disp(sprintf('\tPercentage of values missing:      %5.2f', nmis/(n*p)*100))
  disp(sprintf('\tStagnation tolerance:              %9.2e', stagtol))
  disp(sprintf('\tMaximum number of iterations:     %3i', maxit))
  
  if (inflation ~= 1)
      disp(sprintf('\tResidual (co-)variance inflation:  %6.3f ', inflation))
  end
  
  if Xmis0_given && C0_given
      disp(sprintf(['\tInitialization with given imputed values and' ...
          ' covariance matrix.']))
  elseif C0_given
      disp(sprintf(['\tInitialization with given covariance' ...
          ' matrix.']))
  elseif Xmis0_given
      disp(sprintf('\tInitialization with given imputed values.'))
  else
      disp(sprintf('\tInitialization of missing values by mean substitution.'))
  end
  
  switch regress
      case 'mridge'
          disp(sprintf('\tOne multiple ridge regression per record:'))
          disp(sprintf('\t==> one regularization parameter per record.'))
      case 'iridge'
          disp(sprintf('\tOne individual ridge regression per missing value:'))
          disp(sprintf('\t==> one regularization parameter per missing value.'))
      case 'ttls'
          disp(sprintf('\tOne total least squares regression per record.'))
  end
  
  if D_given
      disp(sprintf('\tUsing scaling for variables given by OPTIONS.scalefac.'))
  else
      disp(sprintf('\tPerforming regularization on correlation matrices.'))
  end
  
  switch regress
      case {'mridge', 'iridge'}
          if regpar_given
              disp(sprintf('\tFixed regularization parameter:    %9.2e', optreg.regpar))
          end
          
      case 'ttls'
          if regpar_given
              disp(sprintf('\tFixed truncation parameter:     %5i', trunc))
          else
              if  strmatch(truncslct, 'kcv', 'exact')
                 disp(sprintf(['\tTruncation choice: K-fold cross validation (K = ', int2str(Kcv), ')']))
                 if ncv < n
                    disp(sprintf(['\tOnly records 1:', int2str(ncv) ' left out in cross-validation.']))
                 end
                 disp(sprintf(['\tNorm of CV error to be minimized: ', int2str(cv_norm)]))
              else
                 disp(sprintf(['\tTruncation choice criterion: ', upper(truncslct)]))
              end
              
              if trial_trunc_given
                 disp(sprintf(['\tTruncation parameters restricted between ', ...
                     int2str(min(trial_trunc)), ' (min) to ', ...
                     int2str(max(trial_trunc)), ' (max).'])) 
              end
              
          end
  end
  
  if Xcmp_given
      disp(sprintf(['\n\tIter \tmean(peff) \t|X-Xcmp|/std(Xcmp) ' ...
          '\t|D(Xmis)|/|Xmis|']))
  else
      disp(sprintf(['\n\tIter \tmean(peff) \t|D(Xmis)| ' ...
          '\t|D(Xmis)|/|Xmis|']))
  end

  % initial estimates of missing values
  if Xmis0_given
    % substitute given guesses for missing values
    X(indmis)  = Xmis0(indmis);
    [X, M]     = center(X);        % center data to mean zero
  else
    [X, M]     = center(X);        % center data to mean zero
    X(indmis)  = zeros(nmis, 1);   % fill missing entries with zeros
  end
      
  if C0_given
    C          = C0;
  else
    C          = X'*X / dofC;      % initial estimate of covariance matrix
  end
  
  % initialize matrices needed for each missingness pattern
  B            = cell(np, 1);
  S            = cell(np, 1);
  for j=1:np
    S{j} = zeros(length(kmisr{j}), length(kmisr{j}));
  end
  peff         = cell(np, 1);
  
  % get cross-validation partitionings for K-fold CV if needed.
  % (Here we fix these partitionings to be invariant from iteration to 
  % iteration, which should help improve convergence. It is also
  % conceivable to use different random partitionings in each iteration.)
  if strmatch(truncslct, 'kcv', 'exact')
      [incv, outcv, nin] = kcvindices(ncv, Kcv, n-ncv);
  end
   
  it           = 0;
  rdXmis       = Inf;
  while (it < maxit && rdXmis > stagtol)
    it         = it + 1;    
    
    % initialize for this iteration ...
    CovRes     = zeros(p, p);      % ... residual covariance matrix
    peff_ave   = 0;                % ... average effective number of variables 

    % scale variables 
    if D_given
        % Strictly speaking, this would not need to be repeated at
        % each iteration when D is given, but easier to code that
        % way. 
      [X, C]     = rescale(X, C, D);  
    else
      [X, C, D]  = rescale(X, C); % scale to unit variance
    end
    
    if strmatch(regress, 'ttls')
      % compute eigendecomposition of covariance/correlation matrix
      [V, d]   = peigs(C, neigs);  
      
      % compute necessary global quantities for truncation selection criteria 
      if strmatch(truncslct, 'kcv', 'exact')
        Xcv = cell(Kcv, 1);
        Mcv = cell(Kcv, 1);
        Ccv = cell(Kcv, 1);
        Dcv = cell(Kcv, 1);
        Vcv = cell(Kcv, 1);
        dcv = cell(Kcv, 1);
        % compute partitionings needed for K-fold CV (this can and should
        % be parallelized to speed it up)
        for k=1:Kcv
                        
          % get CV sample k  
          Xcv{k} = X(incv{k}, :);
          
          % re-center CV sample 
          [Xcv{k}, Mcv{k}] = center(Xcv{k});  
                    
          % compute covariance matrix estimate for CV sample, consisting 
          % of sample covariance matrix of completed CV sample plus 
          % contributions from residual covariance matrices) 
          % first, collect residual covariance contribution
          CovResCv         = zeros(p, p);
          for i=incv{k}
              j            = iptrn(i);
              CovResCv(kmisr{j}, kmisr{j}) = CovResCv(kmisr{j}, kmisr{j}) ...
                  + S{j};
          end
          % rescale residual covariance matrix contribution
          CovResCv         = CovResCv ./ repmat(D', p, 1) ./ repmat(D, 1, p);
          % assemble covariance matrix esimate for CV sample
          Ccv{k}           = (Xcv{k}'*Xcv{k} + CovResCv) ./ (nin{k}-1);
          
          if ~D_given
               % re-standardize data within CV sample
              [Xcv{k}, Ccv{k}, Dcv{k}]  = rescale(Xcv{k}, Ccv{k});
          else
              Dcv{k} = ones(p, 1);
          end
          
          % compute eigendecomposition for CV sample
          neigscv = min([neigs, nin{k}-1, p]);
          [Vcv{k}, dcv{k}] = peigs(Ccv{k}, neigscv);
        end
      else
        % compute global truncation selection criteria
        if ~trial_trunc_given
            trial_trunc = 0 : length(d)-1;
        end
        [mdl, ne08, aic, aicc] = pca_truncation_criteria(d, p, trial_trunc, n);
      end
    end
    
    for j=1:np                     % cycle over missingness patterns  
      pm       = length(kmisr{j}); % number of missing values in this pattern
      if pm > 0	
        pa     = p - pm;           % number of available values in this pattern

        if pa < 1
            error('No available values in at least one row of the data matrix.')
        end
        
        % regression of missing variables on available variables
        switch regress
          case 'mridge'
            % one multiple ridge regression per pattern
            [B{j}, S{j}, ~, peff{j}] = mridge(C(kavlr{j},kavlr{j}), ...
				     C(kmisr{j},kmisr{j}), ...
				     C(kavlr{j},kmisr{j}), n-1, optreg);

            peff{j}  = peff{j} .* ones(pm, 1);
            
          case 'iridge'
            % one individual ridge regression per missing value 
            [B{j}, S{j}, ~, peff{j}] = iridge(C(kavlr{j},kavlr{j}), ...
				     C(kmisr{j},kmisr{j}), ...
				     C(kavlr{j},kmisr{j}), n-1, optreg);
	  
          case 'ttls'
            % truncated total least squares 
            
            if ~regpar_given
                if strmatch(truncslct, 'kcv', 'exact')
                    [trunc, x_rmserr] = kcv_ttls(Vcv, dcv, Dcv, Mcv, kavlr{j}, kmisr{j}, ...
                                    outcv, X, kavlr, iptrn, trial_trunc, cv_norm);
                else
                    imax          = find(trial_trunc <= min(pa, length(d)-1), 1, 'last' );  
                    [~, imin]     = eval(['min(', truncslct, '(1:imax))']);
                    trunc         = trial_trunc(imin);
                end
            end
            
            [B{j}, S{j}]          = pttls(V, d, kavlr{j}, kmisr{j}, trunc);
            peff{j}               = trunc .* ones(pm, 1);
            
        end
	
        % missing value estimates
        Xmis(prows{j}, kmisr{j})  = X(prows{j}, kavlr{j}) * B{j};
        
        % inflation of residual covariance matrix
        S{j}       = inflation * S{j};
        
        % restore original scaling to residual covariance matrix
        S{j}       = S{j} .* repmat(D(kmisr{j})', pm, 1) .* repmat(D(kmisr{j}), 1, pm);
        
        % add up contributions to residual covariance matrix estimate in
        % original scaling
        CovRes(kmisr{j}, kmisr{j}) = CovRes(kmisr{j}, kmisr{j}) + mp(j)*S{j};
        
        peff_ave   = peff_ave + mp(j)*sum(peff{j})/nmis; % add up eff. number of variables
        dofS       = dofC - peff{j};
          
        if strmatch(truncslct, 'kcv', 'exact')
           % use K-fold CV estimate of standard error in imputed values
           Xerr(prows{j}, kmisr{j}) = repmat(x_rmserr .* D(kmisr{j})', mp(j), 1);
        else
           % GCV estimate of standard error in imputed values
           Xerr(prows{j}, kmisr{j}) = repmat(( dofC * sqrt(diag(S{j})) ./ dofS)', mp(j), 1);
        end
      end	
    end                            % loop over patterns
     
    % rescale variables to original scaling 
    X          = X .* repmat(D', n, 1);
    Xmis       = Xmis .* repmat(D', n, 1);
    
    % rms change of missing values
    dXmis      = norm(Xmis(indmis) - X(indmis)) / sqrt(nmis);
    
    % relative change of missing values
    nXmis_pre  = norm(X(indmis) + M(kmis)') / sqrt(nmis);    
    if nXmis_pre < eps
      rdXmis   = Inf;
    else
      rdXmis   = dXmis / nXmis_pre;
    end

    % update data matrix X
    X(indmis)  = Xmis(indmis);
    
    % re-center data and update mean
    [X, Mup]   = center(X);                  % re-center data
    M          = M + Mup;                    % updated mean vector
    
    % update covariance matrix estimate
    C          = (X'*X + CovRes)/dofC; 
      
    if Xcmp_given
      % imputed values in original scaling
      Xmis(indmis) = X(indmis) + M(kmis)'; 
      % relative error of imputed values (relative to values in Xcmp)
      dXmis        = norm( (Xmis(indmis)-Xcmp(indmis))./sXcmp(kmis)' ) ... 
          / sqrt(nmis); 
      disp(sprintf('   \t%3i  \t %8.2e  \t    %10.3e     \t   %10.3e', ...
               it, peff_ave, dXmis, rdXmis))
      
    else
      disp(sprintf('   \t%3i  \t %8.2e  \t%9.3e \t   %10.3e', ...
                it, peff_ave, dXmis, rdXmis))
    end
   
  end                                        % EM iteration

  % add mean back to centered data matrix
  X  = X + repmat(M, n, 1);  
  
  % rescale matrix of regression coefficients so that the matrix output is
  % that for the regression of available values on missing values in the
  % original scaling of variables
  for j=1:np;
    pm     = length(kmisr{j}); % number of missing values in this pattern
    if pm > 0	
      B{j} = diag(1./D(kavlr{j})) * B{j} * diag(D(kmisr{j}));  
    else
      B{j} = NaN;
    end
  end
  % Note that the B matrices are the regression coeffients as of the last
  % iteration of the algorithm, *before the final update of the mean and
  % covariance matrices.* That is, they are the regression coeffients for 
  % the data centered with the penultimate estimate of the mean vector, not 
  % with the final estimate that is returned. With centered data as of the 
  % beginning of the last iteration, 
  %
  % Xc = X - repmat(M-Mup, n, 1);
  %
  % the regression coefficients give the imputed values in the original
  % scaling
  %
  % Xc(prows{j}, kmisr{j}) = Xc(prows{j}, kavlr{j}) * B{j}


