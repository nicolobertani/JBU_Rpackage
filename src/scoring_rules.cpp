#include <RcppArmadillo.h>
using namespace Rcpp;

// RcppArmadillo comments -----------------------
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// ----------------------------------------------


// Scores assume:
// y is matrix of H (forecast horizon) x M (series)
// y_hat is a tensor of N (number samples) x M x H


//' Compute the continuous ranked probability score
//'
//' Computes the continuous ranked probability score (CRPS) for each forecast
//' horizon and series, comparing observed values with simulated forecast samples.
//'
//' @param y A numeric matrix of observed values. Rows are forecast horizons and
//' columns are series.
//' @param y_hat A numeric array of forecast samples with dimensions
//' samples x series x horizons.
//'
//' @return A numeric matrix with the same dimensions as `y`, containing one CRPS
//' value for each horizon and series.
//'
//' @examples
//' set.seed(1)
//' y <- matrix(c(10, 12, 20, 22), nrow = 2, ncol = 2)
//' y_hat <- array(rnorm(5 * 2 * 2, mean = 15, sd = 2), dim = c(5, 2, 2))
//'
//' CRPS_score(y, y_hat)
//'
//' @export
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

//' Compute the energy score
//'
//' Computes the energy score for multivariate probabilistic forecasts at each
//' forecast horizon.
//'
//' @param y A numeric matrix of observed values. Rows are forecast horizons and
//' columns are series.
//' @param y_hat A numeric array of forecast samples with dimensions
//' samples x series x horizons.
//' @param beta A numeric power parameter. The default is `1`.
//'
//' @return A numeric vector with one energy score for each forecast horizon.
//'
//' @examples
//' set.seed(1)
//' y <- matrix(c(10, 12, 20, 22), nrow = 2, ncol = 2)
//' y_hat <- array(rnorm(5 * 2 * 2, mean = 15, sd = 2), dim = c(5, 2, 2))
//'
//' energy_score(y, y_hat)
//'
//' @export
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



//' Compute the variogram score
//'
//' Computes the variogram score for multivariate probabilistic forecasts at
//' each forecast horizon.
//'
//' @param y A numeric matrix of observed values. Rows are forecast horizons and
//' columns are series.
//' @param y_hat A numeric array of forecast samples with dimensions
//' samples x series x horizons.
//' @param p A numeric power parameter. The default is `0.5`.
//'
//' @return A numeric vector with one variogram score for each forecast horizon.
//'
//' @examples
//' set.seed(1)
//' y <- matrix(c(10, 12, 20, 22), nrow = 2, ncol = 2)
//' y_hat <- array(rnorm(5 * 2 * 2, mean = 15, sd = 2), dim = c(5, 2, 2))
//'
//' variogram_score(y, y_hat)
//'
//' @export
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
