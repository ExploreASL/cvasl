is.nan.data.frame <- function(x)
do.call(cbind, lapply(x, is.nan))



#' Estimate sigma
#'
#' This function estimates the standard deviation sigma of the noise of the model where the data are generated from a signal of rank k corrupted by homoscedastic Gaussian noise. 
#' Two estimators are implemented. The first one, named LN, is asymptotically unbiased for sigma in the asymptotic framework where both the number of rows and the number of columns are fixed while the noise variance tends to zero (Low Noise).
#' It is calculated by computing the residuals sum of squares (using the truncated SVD at order k as an estimator) divided by the number of data minus the number of estimated parameters. Thus, it requires as an input the rank k.
#' The second one, MAD (mean absolute deviation) is a robust estimator defined as the ratio of the median of the singular values of X over the square root of the median of the Marcenko-Pastur distribution. It can  be useful when the signal can be considered of low-rank (the rank is very small in comparison to the matrix size).
#' 
#' @param X a data frame or a matrix with numeric entries
#' @param method LN for the low noise asymptotic estimate (it requires to specify the rank k) or MAD for mean absolute deviation
#' @param k integer specifying the rank of the signal only if method = "LN". By default k is estimated using the estim_ncp function of the FactoMineR package 
#' @param  center boolean, to center the data. By default "TRUE". 
#' @return sigma the estimated value
#' @details In the low noise (LN) asymptotic framework,  the estimator requires providing the rank k. Different methods are available in the litterature and if by default the user does not provide any value, we use of the function estim_ncp of the FactoMineR package with the option GCV (see ?estim_ncp).
#' @references Josse, J & Husson, F. (2012). Selecting the number of components in principal component analysis using cross-validation approximations. Computational Statistics & Data Analysis, 6 (56).
#' @references Gavish, M & Donoho, D. L. Optimal Shrinkage of Singular Values.
#' @references Gavish, M & Donoho, D. L. (2014). The Optimal Hard Threshold for Singular Values is 4/sqrt(3). IEEE Transactions on Information Theory, 60 (8), 5040-5053.
#' @references Josse, J. & Husson, F. (2011). Selecting the number of components in PCA using cross-validation approximations.Computational Statististics and Data Analysis. 56 (6), pp. 1869-1879.
#' @seealso \code{\link{estim_ncp}}
#' @seealso \code{\link{LRsim}}
#'  @examples 
#' Xsim <-  LRsim(100, 30, 2, 4)
#' estim_sigma(Xsim$X, 2)


## estim_sigma
estim_sigma <- function(X,
                        k = NA,
                        method = c("LN", "MAD"),
                        center = "TRUE") {
  
  method <- match.arg(method, c("LN","MAD","ln","mad", "Ln","Mad"), several.ok = T)[1]
  method <- tolower(method)
  
  # X <- as.matrix(X)
  
  if(sum(sapply(X, is.numeric)) < ncol(X)){
    stop("all the variables must be numeric")
  }
  
  if(center == "TRUE"){
    X <- scale(X,scale=F)
  }
  
  n = nrow(X) 
  p = ncol(X)
  X[is.nan(X)] <- 0
  X[is.infinite(X)] <- 0
  svdX = svd(X)
  
  # infer unspecified choices
  if(method == "ln" & is.na(k)){
    warning("Since you did not specify k, k was estimated using the FactoMineR estim_ncp function")
    k <- estim_ncp(X, scale = F)$ncp
    print(paste("k = ", k))    
  } 
  
  if(center == "TRUE") {
    N <- (n-1)
  } else {
    N <- n
  }
  
  if((k >= min(N, p))&(method == "ln")){   
    stop("the number k specified has to be smaller than the minimum of the number of rows or columns")
  }
  
  # Begin
  if (method == "ln"){
    
    if(k == 0){
      sigma = sqrt(sum(svdX$d^2)/(N*p))
    } else {                          
      sigma <- sqrt(sum(svdX$d[-c(1:k)]^2)/(N*p  - N*k - p*k + k^2))
    }
  } else {
    beta <- min(n,p)/max(n,p)
    lambdastar <- sqrt( 2*(beta + 1) + 8*beta/((beta + 1 + (sqrt(beta^2 + 14*beta + 1)))))
    wbstar <- 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43
    sigma <-  median(svdX$d)/(sqrt(max(n,p)) *(lambdastar/wbstar))
  }
  
  return(sigma)
}


#' @export
frob=function(X){ sum(X^2,na.rm=T) }

#' @export
sigma.rmt=function(X){ estim_sigma(X,method="MAD") }

#' @export
softSVD=function(X, lambda){
  X[is.nan(X)] <- 0
  X[is.infinite(X)] <- 0
  svdX=svd(X)
  nuc=pmax(svdX$d-lambda,0)
  out=tcrossprod(svdX$u, tcrossprod( svdX$v,diag(nuc) ))
  return(list(out=out, nuc=sum(nuc)))
}

#' @export
relief=function(dat, batch=NULL, mod=NULL,
                scale.features=T, eps=1e-3, max.iter=1000, verbose=T){
  if (verbose) {
    if (!is.null(mod)){
      q=ncol(mod)
      cat(paste0("[RELIEF] Performing RELIEF harmonization with ", ncol(mod), " covariates\n"))
    }
    else{
      q=0
      cat(paste0("[RELIEF] Performing RELIEF harmonization without covariates\n"))
    }
  }
  if (is.null(batch)){ stop("batch information must be provided\n") }
  p=nrow(dat); n=ncol(dat);
  dat.original=dat
  batch.f=as.factor(batch); batch = as.numeric(batch.f)
  batch.id=unique(batch); n.batch=length(batch.id);batch.f.id=unique(batch.f);
  if (verbose) cat(paste0("[RELIEF] ",n.batch," batch identified\n"))
  Xbeta=gamma=sigma.mat=Matrix(0, p, n)
  batch.covariates=model.matrix(~batch.f-1)


  if (is.null(mod)){
    Xbeta = tcrossprod(apply(dat, 1, mean), rep(1,n))
  }else{
    Px= mod%*%ginv(crossprod(mod), tol=0)%*%t(mod)
    Xbeta= dat%*%Px
  }
  residual1 = dat-Xbeta
  Pb = batch.covariates%*%ginv(crossprod(batch.covariates),tol=0)%*%t(batch.covariates)
  gamma = residual1%*%Pb
  residual2 = residual1-gamma

  if (scale.features){
    sigma.mat=sqrt(rowSums(residual2^2)/(n-n.batch-q))%*%t(rep(1,n))
  } else {
    sigma.mat=1
  }

  dat=residual2/sigma.mat
  sub.batch = unlist(lapply(c(1,n.batch), combn, x = batch.id, simplify = FALSE), recursive = FALSE)
  nvec=rep(NA, n.batch)
  sigma.mat.batch=Matrix(1, p, n)
  for (b in 1:n.batch){
    order.temp.batch=which(batch==batch.id[b])
    nvec[b]=length(order.temp.batch)
    

    s=sigma.rmt(dat[, order.temp.batch])

    sigma.mat.batch[, order.temp.batch]=sigma.mat.batch[, order.temp.batch]*s
    dat[, order.temp.batch]=dat[, order.temp.batch]/s
  }

  sigma.harnomized=sqrt(sum((unique(as.numeric(sigma.mat.batch))^2)*nvec)/(sum(nvec)))
  lambda.set=matrix(NA, 1,length(sub.batch))
  for (b in 1:length(sub.batch)){
    lambda.set[1,b]=sqrt(p)+sqrt(sum(nvec[sub.batch[[b]]]))
  }

  index.set.batch = lapply(sub.batch, function(b) which(batch %in% b))

  estim = lapply(1:length(sub.batch), function(x) Matrix(0, p, n, sparse = TRUE))
  bool=TRUE
  count=1; crit0=0
  idx=c(1:length(sub.batch))

  if (verbose) {
    cat(paste0("[RELIEF] Start optimizing...\n"))
    pb = txtProgressBar(min = 0, max=max.iter, initial=0, char="-", style = 3)
  }
  while (bool){
    if (verbose){  setTxtProgressBar(pb, count)  }
    crit0.old = crit0
    nuc.temp=matrix(NA,1, length(sub.batch))
    for (b in length(sub.batch):1){

      temp=softSVD( (dat-Reduce("+", estim[-idx[b]]))[,index.set.batch[[b]]],lambda.set[,b])
      estim[[b]][,index.set.batch[[b]]]=temp$out
      nuc.temp[,b]=temp$nuc
    }

    crit0 = 1/2*frob(dat-Reduce("+", estim))+sum(lambda.set*nuc.temp,na.rm=T)
    if (abs(crit0.old-crit0)<eps){ bool=FALSE }
    else if (count==max.iter){ bool=FALSE}
    else{ count = count+1 }
  }

  if (verbose & count<max.iter){
    setTxtProgressBar(pb, max.iter)
    cat(paste0("\n[RELIEF] Convergence reached. Finish harmonizing.\n"))
  }
  if (verbose & count==max.iter){
    cat(paste0("\n[RELIEF] Convergence not reached. Increase max.iter.\n"))
  }

  E=dat-Reduce("+", estim)
  E.scaled=sigma.mat*E
  E.original=sigma.mat*sigma.mat.batch*E
  R=sigma.mat*sigma.mat.batch*estim[[length(index.set.batch)]]
  I=sigma.mat*sigma.mat.batch*Reduce("+", estim[-length(index.set.batch)])
  harmonized=Xbeta+R+sigma.harnomized*E.scaled
  estimates=list(Xbeta=Xbeta,gamma=gamma,sigma.mat=sigma.mat, sigma.mat.batch=as.matrix(sigma.mat.batch),sigma.harnomized=sigma.harnomized, R=as.matrix(R),I=as.matrix(I),E.scaled=as.matrix(E.scaled), E.original=as.matrix(E.original))

  return(list(dat.relief=as.matrix(harmonized),
              estimates=estimates,dat.original=dat.original,
              batch=batch.f))
}
