#include <RcppArmadillo.h>
using namespace Rcpp;

// RcppArmadillo comments -----------------------
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// ----------------------------------------------


// Scores assume:
// y is matrix of H (forecast horizon) x M (series)
// y_hat is a tensor of N (number samples) x M x H


// CRPS
// [[Rcpp::export]]
arma::mat CRPS_score (arma::mat y, arma::cube y_hat) {
  // checks
  int H = y.n_rows;
  if (H != y_hat.n_slices) {
    Rcpp::stop("forecast horizon does not match!");
  }
  int M = y.n_cols;
  if (M != y_hat.n_cols) {
    Rcpp::stop("number of series does not match!");
  }
  int N = y_hat.n_rows;

  // score
  arma::mat out(H, M);
  for (arma::uword m = 0; m < M; m++) {
    for (arma::uword h = 0; h < H; h++) {
      double score = 0;
#pragma omp parallel for reduction(+:score)
      for (arma::uword i = 0; i < N; i++) {
        arma::vec y_hat_mh = y_hat(0, m, h, arma::size(N, 1, 1));
        score += std::abs(y(h, m) - y_hat(i, m, h))
          - .5 * arma::mean(arma::abs(y_hat(i, m, h) - y_hat_mh));
      }
      out(h, m) = score / N;
    }
  }
  return out;
}

// Energy score
// [[Rcpp::export]]
arma::vec energy_score (arma::mat y, arma::cube y_hat, double beta = 1) {
  // checks
  int H = y.n_rows;
  if (H != y_hat.n_slices) {
    Rcpp::stop("forecast horizon does not match!");
  }
  int M = y.n_cols;
  if (M != y_hat.n_cols) {
    Rcpp::stop("number of series does not match!");
  }
  int N = y_hat.n_rows;

  // score
  arma::vec out(H);
  for (arma::uword h = 0; h < H; h++) {
    double score = 0;
    arma::vec y_h = y.row(h).t();
    arma::mat y_hat_h = y_hat.slice(h);
#pragma omp parallel for reduction(+:score)
    for (arma::uword i = 0; i < N; i++) {
      arma::vec y_hat_hi = y_hat_h.row(i).t();
      score += std::pow(arma::norm(y_h - y_hat_hi), beta) / N; // part 1
      for (arma::uword j = 0; j < N; j++) {
        arma::vec y_hat_hj = y_hat_h.row(j).t();
        score -= std::pow(arma::norm(y_hat_hi - y_hat_hj), beta) / (2 * std::pow(N, 2)); // part 2
      }
    }
    out(h) = score;
  }
  return out;
}



// Variogram
// [[Rcpp::export]]
arma::vec variogram_score (arma::mat y, arma::cube y_hat, double p = .5) {
  // checks
  int H = y.n_rows;
  if (H != y_hat.n_slices) {
    Rcpp::stop("forecast horizon does not match!");
  }
  int M = y.n_cols;
  if (M != y_hat.n_cols) {
    Rcpp::stop("number of series does not match!");
  }

  // score
  arma::vec out(H);
  for (arma::uword h = 0; h < H; h++) {
    double score = 0;
    arma::vec y_h = y.row(h).t();
    arma::mat y_hat_h = y_hat.slice(h);
    // #pragma omp parallel for reduction(+:score)
    for (arma::uword m_i = 0; m_i < M; m_i++) {
      for (arma::uword m_j = 0; m_j < M; m_j++) {
        score += std::pow(
          std::pow(std::abs(y_h(m_i) - y_h(m_j)), p)
        - arma::mean(arma::pow(arma::abs(y_hat_h.col(m_i) - y_hat_h.col(m_j)), p))
        , 2);
      }
    }
    out(h) = score / std::pow(M, 2);
  }
  return out;
}
