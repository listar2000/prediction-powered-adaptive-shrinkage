library(MASS)
library(ggplot2)
library(reshape2)
library(knitr)
library(progress)
library(REBayes)
library(stringr)

library(hdf5r)
library(rhdf5)



.compute_summary <- function(Y, Z, r, seed = NULL) {
  Tn <- length(Y)
  if (!is.null(seed)) set.seed(seed)
  
  theta_true <- mean(Y)
  
  # 1. Split Data
  n_lab <- max(1L, min(Tn - 1L, floor(r * Tn)))
  lab_i  <- sample.int(Tn, n_lab)
  unlab_i <- setdiff(seq_len(Tn), lab_i)
  n_unl <- length(unlab_i)
  
  y_lab <- Y[lab_i]
  z_lab <- Z[lab_i]
  z_unl <- Z[unlab_i]
  
  # 2. Point Estimates (Means)
  Z_tilde <- mean(z_unl)                          
  Z_bar   <- mean(z_lab)                          
  Y_bar   <- mean(y_lab)                          
  
  # Original "Vanilla" Estimators (Lambda=1 implicitly)
  b_hat     <- Y_bar - Z_bar
  theta_hat <- Z_tilde + b_hat
  
  # 3. Variance Components needed for Inference (Local Split)
  # Standard Errors for the specific labeled/unlabeled sets
  sZ2_unl <- if (n_unl >= 2) var(z_unl) else NA_real_ # Var of Z on Unlabeled
  sY2_lab <- if (n_lab >= 2) var(y_lab) else NA_real_ # Var of Y on Labeled
  sZ2_lab <- if (n_lab >= 2) var(z_lab) else NA_real_ # Var of Z on Labeled
  cov_lab <- if (n_lab >= 2) cov(y_lab, z_lab) else NA_real_ # Cov on Labeled
  sR2     <- if (n_lab >= 2) var(y_lab - z_lab) else NA_real_ # Var of Vanilla Residual
  
  # Calculate Standard Errors (SE^2)
  var_b      <- if (!is.na(sR2)) sR2 / n_lab else NA_real_
  var_Ztilde <- if (!is.na(sZ2_unl)) sZ2_unl / n_unl else NA_real_
  
  # Vanilla Variance of Theta
  var_theta  <- if (is.na(var_b) | is.na(var_Ztilde)) NA_real_ else (ifelse(is.na(var_b), 0, var_b) + ifelse(is.na(var_Ztilde), 0, var_Ztilde))
  var_Ybar   <- if (!is.na(sY2_lab)) sY2_lab / n_lab else NA_real_
  
  corr_theta_b <- if (!is.na(var_theta) && !is.na(var_b) && var_theta > 0 && var_b >= 0) {
    sqrt(var_b / var_theta)
  } else NA_real_
  
  # 4. STATS FOR UNIPT (Global & Reconstruction)
  # We need Var(f) using ALL data to get the best global Lambda estimate
  var_f_global <- var(Z) 
  
  data.frame(
    theta_true,
    
    # --- Sample Sizes ---
    n_total      = Tn,
    n_labeled    = n_lab,
    n_unlabeled  = n_unl,
    
    # --- Estimators ---
    Z_tilde      = Z_tilde,
    Z_bar        = Z_bar,
    Y_bar        = Y_bar,
    b_hat        = b_hat,
    theta_hat    = theta_hat,
    
    # --- Variances (Vanilla) ---
    var_theta_hat= var_theta,
    var_b_hat    = var_b,
    corr_theta_b = corr_theta_b,
    var_Z_tilde  = var_Ztilde,
    var_Y_bar    = var_Ybar,
    
    # --- Stats for UniPT Calculation ---
    var_f_global = var_f_global,  # Global Var(Z) -> For Lambda Denominator
    var_z_unlab    = sZ2_unl,
    var_z_lab    = sZ2_lab,       # Local Var(Z)  -> For Local Residual Update
    var_y_lab    = sY2_lab,       # Local Var(Y)  -> For Local Residual Update
    cov_yf_lab   = cov_lab,       # Local Cov(Y,Z)-> For Lambda Numerator & Update
    
    stringsAsFactors = FALSE
  )
}

calculate_unipt_and_update <- function(sum_stat) {
  
  library(dplyr)
  
  # -------------------------------------------------------------------------
  # STEP 1: Calculate Universal Lambda (Same as before)
  # -------------------------------------------------------------------------
  
  lambda_num <- sum(sum_stat$cov_yf_lab / sum_stat$n_labeled, na.rm = TRUE)
  weights <- (1/sum_stat$n_labeled) + (1/sum_stat$n_unlabeled)
  lambda_denom <- sum(sum_stat$var_f_global * weights, na.rm = TRUE)
  lambda_uni <- lambda_num / lambda_denom
  
  # -------------------------------------------------------------------------
  # STEP 2: Add New Columns (Prefix 'lambda_')
  # -------------------------------------------------------------------------
  
  updated_stat <- sum_stat %>%
    mutate(
      # --- 0. Store the Global Lambda Value ---
      lambda_value = lambda_uni,
      
      # --- A. New Estimators ---
      # New Bias: Y - lam*Z_bar
      lambda_b_hat = Y_bar - (lambda_uni * Z_bar),
      
      # New Theta: lambda_b + lam*Z_tilde
      lambda_theta_hat = lambda_b_hat + (lambda_uni * Z_tilde),
      
      # --- B. New Variances (Intermediate Calcs hidden) ---
      
      # 1. Variance of the New Residual (on labeled set)
      # Var(Y - lam*Z) = Var(Y) + lam^2*Var(Z) - 2*lam*Cov(Y,Z)
      temp_var_resid_local = var_y_lab + (lambda_uni^2 * var_z_lab) - (2 * lambda_uni * cov_yf_lab),
      
      # Final Var(b) = Var(Residual) / n
      lambda_var_b_hat = temp_var_resid_local / n_labeled,
      
      # 2. Variance of the Scaled Unlabeled Prediction
      # Var(lam * Z_tilde) = lam^2 * Var(Z_tilde)
      temp_var_pred_unl = lambda_uni^2 * var_Z_tilde,
      
      # 3. Variance of Theta
      # Var(Theta) = Var(b) + Var(Scaled_Pred)
      lambda_var_theta_hat = lambda_var_b_hat + temp_var_pred_unl,
      
      # 4. Correlation
      # sqrt(Var(b) / Var(Theta))
      lambda_corr_theta_b = sqrt(lambda_var_b_hat / lambda_var_theta_hat)
    ) %>%
    # Remove the temporary intermediate calculation columns
    select(-temp_var_resid_local, -temp_var_pred_unl)
  
  return(updated_stat)
}

load_data_amazon <- function(file, r, seed = NULL) {
  stopifnot(file.exists(file), r > 0, r < 1)
  ls_tbl <- rhdf5::h5ls(file, recursive = FALSE)
  keys   <- ls_tbl$name[ls_tbl$otype == "H5I_DATASET"]
  if (length(keys) == 0L) stop("No root-level datasets found.")
  
  rows <- lapply(keys, function(k) {
    m <- rhdf5::h5read(file, paste0("/", k))  # exactly your call
    if (!is.matrix(m) || ncol(m) < 2)
      stop(sprintf("Dataset '%s' is not a 2-column matrix.", k))
    Y <- as.numeric(m[, 1])
    Z <- as.numeric(m[, 2])
    cbind(data.frame(group = k, stringsAsFactors = FALSE),
          .compute_summary(Y, Z, r = r, seed = seed))
  })
  
  do.call(rbind, rows)
}




#----Metric----
# 1) empirical coverage
ci_coverage <- function(ci_mat, theta_true) {
  stopifnot(all(c("lower", "upper") %in% colnames(ci_mat)))
  mean(theta_true >= ci_mat[, "lower"] & theta_true <= ci_mat[, "upper"])
}

# 2) average interval length
ci_avg_length <- function(ci_mat) {
  stopifnot(all(c("lower", "upper") %in% colnames(ci_mat)))
  mean(ci_mat[, "upper"] - ci_mat[, "lower"])
}

# length for each row
ci_len_vec <- function(ci_mat) {
  stopifnot(all(c("lower","upper") %in% colnames(ci_mat)))
  ci_mat[, "upper"] - ci_mat[, "lower"]
}

# mean of per-item ratios: mean( L_method_i / L_ref_i )
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


