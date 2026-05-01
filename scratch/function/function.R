library(MASS)
library(ggplot2)
library(reshape2)
library(progress)
library(REBayes)
library(latex2exp)


# ----BH-procedure for simulation----
BH_adjust <- function(p_values, alpha = 0.05){
  sorted_indices <- order(p_values)
  sorted_p_values <- p_values[sorted_indices]
  bh_adjusted_p <- p.adjust(sorted_p_values, method = "BH")
  bh_adjusted_p_original_order <- bh_adjusted_p[order(sorted_indices)]
  significance <- bh_adjusted_p_original_order <= alpha
  return(significance)
}



# ----Prior Estimation----
GNmix_diffvar <- function(hat_delta_g_set,se_delta_set,u = 30, threshold=1e-06,verbose = TRUE,...){
  t <- hat_delta_g_set
  G <- length(t)
  w <- rep(1, G)/G
  eps <- 1e-05
  
  if (length(u) == 1) 
    u <- seq(min(t)-eps, max(t)+eps, length = u)
  
  pu <- length(u)
  du <- rep(1, pu)
  
  if (verbose) {cat("Calculating the constraint matrix:")}
  start_time_calculation <- Sys.time()
  Au <- matrix(NA, G, pu)
  for (i in 1:G){
    for (j in 1:pu){
      Au[i,j] = dnorm((t[i]-u[j])/se_delta_set[i])/se_delta_set[i]
    }
  }
  end_time_calculation <- Sys.time()
  if (verbose){cat(sprintf(" done. Execution time: %.2f minutes\n", difftime(end_time_calculation, start_time_calculation, units = "mins")))}
  
  start_time_mosek <- Sys.time()
  if (verbose){cat("Solving for discretized NPMLE:")}
  f <- KWDual(Au, du, w,...)
  end_time_mosek <- Sys.time()
  execution_time_mosek <- end_time_mosek-start_time_mosek
  if (verbose){cat(sprintf(" done. Execution time: %.2f minutes\n", difftime(end_time_mosek, start_time_mosek, units = "mins")))}
  fu <- f$f
  z <- list(u = u, fu = fu,status = f$status)
  class(z) <- "GNmix"
  return(z)
}

estimate_parametric_mle <- function(hat_b, se_hat_b, init_mean = NULL, init_var = NULL) {
  # hat_b: numeric vector of observed \hat{b}_i
  # se_hat_b: same length, measurement SE for each \hat{b}_i
  stopifnot(length(hat_b) == length(se_hat_b))
  m <- length(hat_b)
  
  # start values (method of moments style) if not given
  if (is.null(init_mean)) init_mean <- mean(hat_b)
  if (is.null(init_var)) {
    # crude: sample var minus avg meas var, truncated at small positive
    s2_obs <- var(hat_b)
    avg_meas <- mean(se_hat_b^2)
    init_var <- max(s2_obs - avg_meas, 1e-4)
  }
  
  # negative log-likelihood
  nll <- function(par) {
    mu    <- par[1]
    log_v <- par[2]
    v     <- exp(log_v)   # this is b0 (the prior variance), >0
    
    # total var for obs i
    tot_var <- v + se_hat_b^2
    # log density of normal
    # -0.5 * [log(2*pi*tot_var) + (x-mu)^2 / tot_var]
    resid2 <- (hat_b - mu)^2
    val <- 0.5 * sum(log(2*pi*tot_var) + resid2 / tot_var)
    val
  }
  
  opt <- optim(
    par = c(init_mean, log(init_var)),
    fn  = nll,
    method = "BFGS"
  )
  
  mu_hat    <- opt$par[1]
  var_hat   <- exp(opt$par[2])
  
  list(
    hat_mu_b = mu_hat,      # estimated prior mean
    hat_var_b = var_hat,     # estimated prior variance
    loglik = -opt$value,
    convergence = opt$convergence
  )
}

# ----P-value calculation----
cal_p_value_z_diffvar <- function(hat_beta_g,se_beta){
  p_value <- 2*pnorm(-abs(hat_beta_g)/se_beta)
  return(p_value)
}


cal_p_value_npmle_diffvar <- function(value,probability,hat_beta_g,hat_delta_g,se_beta,se_delta,corr){
  
  temp <- hat_delta_g - value
  Sigma <- matrix(
    c(se_beta^2, corr * se_beta * se_delta,
      corr * se_beta * se_delta, se_delta^2),
    nrow = 2, byrow = TRUE
  )
  Omega <- solve(Sigma)
  term1 <- (temp * Omega[1, 2])^2 / as.numeric(2 * Omega[1, 1])
  term2 <- temp^2 * Omega[2, 2] / 2
  
  prob <- probability * exp(term1 - term2)
  k_prime <- prob / sum(prob)
  
  cond_mean <- -temp * Omega[1, 2] / as.numeric(Omega[1, 1])
  cond_sd <- sqrt(1 / as.numeric(Omega[1, 1]))
  
  F_obs <- sum(k_prime * pnorm(hat_beta_g, mean = cond_mean, sd = cond_sd))
  
  p_value <- 2 * min(F_obs, 1 - F_obs)
  
  return(p_value)
}

cal_p_value_npmle_diffvar_abs <- function(value,probability,hat_beta_g,hat_delta_g,se_beta,se_delta,corr){
  I <- length(probability)
  temp <- hat_delta_g - value
  Sigma <- matrix(c(se_beta**2, corr*se_beta*se_delta, corr*se_beta*se_delta, se_delta**2), nrow=2, byrow=TRUE)
  Omega <- solve(Sigma)
  
  term1 <- (temp*Omega[1,2]) ^ 2 / as.numeric(2 *Omega[1,1])
  term2 <- temp**2*Omega[2,2]/2
  prob <- probability * exp(term1 - term2)
  k_prime <- prob/sum(prob)
  mean <- -temp * Omega[1,2] / as.numeric(Omega[1,1])
  sd <- sqrt(1 /as.numeric(Omega[1,1]))
  p_value <- sum(k_prime *  ( pnorm(-abs(hat_beta_g) - mean, sd = sd) + pnorm(-abs(hat_beta_g) + mean, sd = sd)))
  return(p_value)
}

cal_p_value_param_diffvar <- function(hat_mu_b, hat_var_b, hat_beta_g, hat_delta_g, se_beta, se_delta, corr){
  
  var_beta  <- se_beta^2
  var_delta <- se_delta^2
  cov_bd    <- corr * se_beta * se_delta
  
  var_hat_delta_marginal <- var_delta + hat_var_b
  cond_mean <- cov_bd / var_hat_delta_marginal * (hat_delta_g - hat_mu_b)
  
  cond_var <- var_beta - cov_bd^2 / var_hat_delta_marginal
  cond_sd <- sqrt(cond_var)
  
  F_obs <- pnorm(hat_beta_g, mean = cond_mean, sd = cond_sd)
  p_value <- 2 * pmin(F_obs, 1 - F_obs)
  p_value <- pmin(p_value, 1)
  
  return(p_value)
}


# ----Plot----
average_marginal_delta_npmle <- function(x, value, probability, se_delta_set) {
  # For each gene, compute the marginal density at x
  G <- length(se_delta_set)
  marginal_vals <- sapply(se_delta_set, function(se) {
    sum(dnorm(x, mean = value, sd = se) * probability)
  })
  # Average over all genes
  mean(marginal_vals)
}

average_marginal_delta_parametric <- function(x, a, b, se_delta_set) {
  # x: point at which to evaluate the marginal density
  # a, b: prior delta ~ N(a, b)  (b is the prior variance)
  # se_delta_set: vector of standard errors (one per group)
  
  marginal_vals <- sapply(se_delta_set, function(se) {
    dnorm(x, mean = a, sd = sqrt(b + se^2))
  })
  mean(marginal_vals)
}



# ----Confidence interval----
##----CI with parametric prior----
cal_ci_parametric <- function(a, b,
                              hat_beta_g, hat_delta_g,
                              se_beta, se_delta, rho,
                              alpha = 0.05) {

  sigma_beta  <- se_beta
  sigma_delta <- se_delta
  sigma_bd    <- rho * se_beta * se_delta   # cov(hat_beta, hat_delta)
  
  k <- sigma_bd / (sigma_delta^2 + b)
  
  beta_hat <- hat_beta_g - k * (hat_delta_g - a)
  
  var_beta_hat <- sigma_beta^2 - sigma_bd^2 / (sigma_delta^2 + b)
  sd_beta_hat  <- sqrt(var_beta_hat)
  
  z <- qnorm(1 - alpha/2)
  
  c(
    lower = beta_hat - z * sd_beta_hat,
    upper = beta_hat + z * sd_beta_hat
  )
}

##----CI with NPMLE prior----
.npmle_mixture_params <- function(value, probability,
                                  hat_delta_g, se_beta, se_delta, corr) {
  Sigma <- matrix(c(se_beta^2, corr*se_beta*se_delta,
                    corr*se_beta*se_delta, se_delta^2), 2, 2, byrow = TRUE)
  Omega <- solve(Sigma)
  
  temp <- hat_delta_g - value  # vector
  term1 <- (temp * Omega[1,2])^2 / as.numeric(2 * Omega[1,1])
  term2 <- (temp^2) * Omega[2,2] / 2
  w_unnorm <- probability * exp(term1 - term2)
  w <- as.numeric(w_unnorm / sum(w_unnorm))
  
  mu <- as.numeric(-temp * Omega[1,2] / Omega[1,1])
  sd <- rep(sqrt(1 / as.numeric(Omega[1,1])), length(mu))
  
  list(w = w, mu = mu, sd = sd)
}
# ---- CDF and quantile of a Normal mixture  sum w_m N(mu_m, sd_m^2)
.cdf_mix <- function(x, pars) {
  # pars: list(w, mu, sd)
  sum(pars$w * pnorm((x - pars$mu) / pars$sd))
}

.q_mix <- function(p, pars) {
  lo <- min(pars$mu - 10 * pars$sd)
  hi <- max(pars$mu + 10 * pars$sd)
  uniroot(function(z) .cdf_mix(z, pars) - p, interval = c(lo, hi))$root
}

cal_ci_npmle <- function(value, probability,
                         hat_beta_g, hat_delta_g,
                         se_beta, se_delta, corr,
                         alpha = 0.05) {
  
  pars <- .npmle_mixture_params(value, probability, hat_delta_g,
                                se_beta, se_delta, corr)
  # Quantiles of (hat_beta - beta) | hat_delta  ~ mixture
  q_lo  <- .q_mix(alpha/2,       pars)  
  q_hi  <- .q_mix(1 - alpha/2,   pars)   
  
  # Invert to get CI 
  ci <- c(lower = hat_beta_g - q_hi, upper = hat_beta_g - q_lo)
  return(ci)
}

##----CI for ppi/classic estimator----
cal_ci_normal <- function(est, se, alpha = 0.05) {
  stopifnot(alpha > 0, alpha < 1)
  z <- qnorm(1 - alpha / 2)
  c(lower = est - z * se, upper = est + z * se)
}

#----Metric----

ci_coverage <- function(ci_mat, theta_true) {
  stopifnot(all(c("lower", "upper") %in% colnames(ci_mat)))
  mean(theta_true >= ci_mat[, "lower"] & theta_true <= ci_mat[, "upper"])
}

ci_avg_length <- function(ci_mat) {
  stopifnot(all(c("lower", "upper") %in% colnames(ci_mat)))
  mean(ci_mat[, "upper"] - ci_mat[, "lower"])
}


ci_mean_ratio <- function(ci_mat, ci_ref) {
  L  <- ci_len_vec(ci_mat)
  L0 <- ci_len_vec(ci_ref)
  # guard against zeros to avoid Inf/NaN
  ok <- is.finite(L) & is.finite(L0) & (L0 > 0)
  mean(L[ok] / L0[ok])
}

#coverage rate for one CI matrix (n x 2)
cov_rate <- function(ci_mat, theta) {
  stopifnot(nrow(ci_mat) == length(theta))
  lo <- pmin(ci_mat[,1], ci_mat[,2])
  hi <- pmax(ci_mat[,1], ci_mat[,2])
  mean(theta >= lo & theta <= hi, na.rm = TRUE)
}


