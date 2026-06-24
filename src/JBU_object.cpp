#include "progress_bar.h"
#include "sampling_functions.h"
#include <RcppArmadillo.h>
#include <chrono>
#include <iostream>

using namespace Rcpp;
using namespace arma;

// RcppArmadillo comments -----------------------
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// ----------------------------------------------

// JBU_model class
class JBU_model {
private:

  arma::mat m_X_block, m_y_mx;
  arma::umat m_delta;
  arma::sp_mat m_X_alpha;
  int m_M, m_T;
  // hyperparameters
  double m_psi = pow(10, 4); // beta
  double m_a_1 = 5; // tau
  double m_a_2 = 50; // tau
  double m_epsilon = .005; // gamma
  double m_w = .5; // omega
  double m_v_0;
  arma::mat m_S_0;
  // sampler draws
  arma::vec m_kappa_draw, m_inv_tau_sq_draw, m_beta_draw;
  arma::mat m_inv_Sigma, m_GRR_mx, m_Sigma, m_chol_U_inv_Sigma;
  // sampler storage - average
  double m_w_avg;
  arma::vec m_kappa_avg, m_inv_tau_sq_avg, m_beta_avg;
  arma::mat m_Sigma_avg;
  // sampler storage - distributions
  arma::vec m_w_list;
  arma::mat m_kappa_list, m_tau_sq_list, m_beta_list;
  arma::cube m_Sigma_list;
  // sampler storage - posterior sample
  arma::cube m_post_sample;

  // FILTER
  /* calculate individual approximate posterior probability
  to be used in the loop in the filter */
  double calculate_pip (const arma::uword &i, const arma::uword &j, const double &yTy) {
    double inv_A_N, A_N, a_N, B;
    inv_A_N = dot(m_X_block.col(j), m_X_block.col(j)) + 1 / m_psi;
    A_N = 1 / inv_A_N;
    a_N = A_N * dot(m_X_block.col(j), m_y_mx.col(i));
    B = .5 * (yTy - pow(a_N, 2) * inv_A_N);
    return log(A_N) - (m_X_block.n_rows - 1) * log(B);
  }

  double calculate_sequential_pip (const arma::uword &i, const arma::uvec &active_v, const double &yTy) {
    double B;
    arma::mat inv_A_0, inv_A_N, A_N;
    arma::vec a_N;
    arma::mat X_active = m_X_block.cols(active_v);
    arma::vec y_i = m_y_mx.col(i);
    arma::vec inv_A_0_v(active_v.n_elem);
    inv_A_0_v.fill(1 / m_psi);
    inv_A_0 = arma::diagmat(inv_A_0_v);
    inv_A_N = X_active.t() * X_active + inv_A_0;
    A_N = arma::inv_sympd(inv_A_N);
    a_N = A_N * X_active.t() * y_i;
    B = .5 * arma::as_scalar(y_i.t() * y_i - a_N.t() * inv_A_N * a_N);
    return std::log(arma::det(A_N)) - (m_X_block.n_rows - 1) * std::log(B);
  }

  // SAMPLERS
  void set_Wishart_hp (const bool &RATS) {
    if (RATS) {
      m_v_0 = m_y_mx.n_cols * (m_delta.n_rows + 1) - 2; //RATS
    } else {
      m_v_0 = m_y_mx.n_cols + 1; // Jeffrey's
    }
    m_S_0.eye(m_y_mx.n_cols, m_y_mx.n_cols);
    m_S_0 = m_S_0 * m_v_0;
  }
  void set_initial_draws () {
    m_beta_draw.set_size(m_delta.n_elem);
    m_beta_draw.zeros();
    m_kappa_draw.set_size(m_delta.n_elem);
    m_kappa_draw.ones();
    m_inv_tau_sq_draw.set_size(m_delta.n_elem);
    m_inv_tau_sq_draw.fill(1 / pow(10, 6));
    m_inv_Sigma.set_size(m_y_mx.n_cols, m_y_mx.n_cols);
    m_inv_Sigma.eye();
  }
  void initialize_avg () {
    m_w_avg = 0;
    m_beta_avg.copy_size(m_beta_draw);
    m_beta_avg.zeros();
    m_kappa_avg.copy_size(m_kappa_draw);
    m_kappa_avg.zeros();
    m_inv_tau_sq_avg.copy_size(m_inv_tau_sq_draw);
    m_inv_tau_sq_avg.zeros();
    m_Sigma_avg.copy_size(m_inv_Sigma);
    m_Sigma_avg.zeros();
  }
  void initialize_lists (const int &iterations, const int &burn) {
    int n = iterations - burn;
    m_w_list.set_size(n);
    m_beta_list.set_size(m_delta.n_elem, n);
    m_kappa_list.set_size(m_delta.n_elem, n);
    m_tau_sq_list.set_size(m_delta.n_elem, n);
    m_Sigma_list.set_size(m_y_mx.n_cols, m_y_mx.n_cols, n);
  }
  void initialize_post_sample (const int &iterations, const int &burn, const int &step_ahead) {
    m_post_sample.set_size(step_ahead, m_y_mx.n_cols, iterations - burn);
  }
  // update functions
  void beta_update(const mat &Kron_hlpr, const vec &y_vec, const double &blocks) {
    mat chol_U_Inv_Sigma = kron(m_chol_U_inv_Sigma, Kron_hlpr);
    mat UX = chol_U_Inv_Sigma * m_X_alpha;
    vec U_y_vec = chol_U_Inv_Sigma * y_vec;
    double q = ceil(m_delta.n_elem / blocks);
    for (uword j = 0; j < blocks; j++) {
      mat UX_j, R_mx_j, inv_B_N_j, B_N_j, UX_shed;
      vec b_N_j, beta_draw_shed, end_vec(2);
      // create selectors for block
      int start = j * q;
      end_vec(0) = (j + 1) * q;
      end_vec(1) = m_delta.n_elem;
      int end = end_vec.min() - 1;
      // block quantities and sample
      UX_j = UX.cols(start, end);
      R_mx_j = m_GRR_mx.submat(start, start, end, end);
      inv_B_N_j = R_mx_j + UX_j.t() * UX_j;
      B_N_j = inv_sympd(inv_B_N_j);
      UX_shed = UX;
      UX_shed.shed_cols(start, end);
      beta_draw_shed = m_beta_draw;
      beta_draw_shed.shed_rows(start, end);
      b_N_j = B_N_j * UX_j.t() * (U_y_vec - UX_shed * beta_draw_shed);
      // draw
      m_beta_draw.subvec(start, end) = mvnrnd(b_N_j, B_N_j);
    }
  }
  void kappa_update () {
    vec beta_sq, sd_term, w_1, w_2, v;
    beta_sq = pow(m_beta_draw, 2);
    sd_term = .5 * m_inv_tau_sq_draw % beta_sq;
    w_1 = exp(log(1 - m_w) - .5 * log(m_epsilon) - sd_term / m_epsilon);
    w_2 = exp(log(m_w) - sd_term);
    // draw
    v = r_binom(m_beta_draw.n_elem, w_1 / (w_1 + w_2));
    m_kappa_draw = m_epsilon * v + (1 - v);
  }
  void inv_tau_sq_update () {
  vec beta_sq, a_N_2;
  beta_sq = pow(m_beta_draw, 2);
  a_N_2 = m_a_2 + .5 * beta_sq / m_kappa_draw;
  // draw
  m_inv_tau_sq_draw = r_gamma(m_inv_tau_sq_draw.n_elem, m_a_1 + .5, a_N_2);
}
  void w_update() {
    double v, V;
    v = 1 + sum(m_kappa_draw == 1);
    V = 1 + sum(m_kappa_draw != 1);
    // draw
    m_w = r_beta(v, V);
  }
  void GRR_update() { m_GRR_mx = diagmat(m_inv_tau_sq_draw / m_kappa_draw); }
  void inv_Sigma_update (const vec &y_vec) {
    double v_N;
    vec r_vec;
    mat S_N, inv_S_N, r_mx;
    r_vec = y_vec - m_X_alpha * m_beta_draw;
    r_mx.set_size(m_T, m_M);
    for (uword i = 0; i < m_y_mx.n_cols; i++) {
      r_mx.col(i) = r_vec.subvec(i * m_X_block.n_rows, (i + 1) * m_X_block.n_rows - 1);
    }
    v_N = m_v_0 + m_T;
    S_N = m_S_0 + r_mx.t() * r_mx;
    inv_S_N = inv_sympd(S_N);
    // draw
    m_inv_Sigma = arma::wishrnd(inv_S_N, v_N);
  }
  void compute_chol_inv_Sigma() {m_chol_U_inv_Sigma = chol(m_inv_Sigma);}
  void compute_Sigma() {
    mat chol_U_Sigma = inv(trimatu(m_chol_U_inv_Sigma));
    m_Sigma = chol_U_Sigma * chol_U_Sigma.t();
  }

  // STORE SAMPLES
  void update_avg() {
    m_beta_avg += m_beta_draw;
    m_kappa_avg += m_kappa_draw;
    m_inv_tau_sq_avg += m_inv_tau_sq_draw;
    m_w_avg += m_w;
    m_Sigma_avg += m_Sigma;
  }
  void update_lists(const int &iter, const int &burn) {
    uword n = iter - burn;
    m_w_list(n) = m_w;
    m_beta_list.col(n) = m_beta_draw;
    m_kappa_list.col(n) = m_kappa_draw;
    m_tau_sq_list.col(n) = 1 / m_inv_tau_sq_draw;
    m_Sigma_list.slice(n) = m_Sigma;
  }
  void update_post_sample(const int &iter, const int &burn, const int &step_ahead) {
    uword n = iter - burn;
    int lags = m_X_block.n_cols / m_y_mx.n_cols;
    mat X_pred(lags + step_ahead, m_y_mx.n_cols, fill::zeros);
    X_pred.head_rows(lags) = m_y_mx.tail_rows(lags);
    for (uword i = 0; i < step_ahead; i++) {
      sp_mat X_tmp(m_delta.n_rows * m_y_mx.n_cols, m_y_mx.n_cols);
      vec x_tmp = vectorise(reverse(X_pred.rows(i, i + lags - 1), 0));
      for (uword j = 0; j < m_y_mx.n_cols; j++) {
        X_tmp.submat(j * m_delta.n_rows, j, (j + 1) * m_delta.n_rows - 1, j) = x_tmp.elem(m_delta.col(j));
      }
      X_pred.row(lags + i) = mvnrnd(X_tmp.t() * m_beta_draw, m_Sigma).t();
    }
    m_post_sample.slice(n) = X_pred.tail_rows(step_ahead);
  }
  void normalize_avg(const int &iterations, const int &burn) {
    double n = iterations - burn;
    m_beta_avg /= n;
    m_kappa_avg /= n;
    m_inv_tau_sq_avg /= n;
    m_w_avg /= n;
    m_Sigma_avg /= n;
  }


public:

  void set_slab_parameters (const double & a_1, const double & a_2) {
    m_a_1 = a_1;
    m_a_2 = a_2;
  }

  // FILTER
  // construct and print X and y
  void set_X_block (mat value) { m_X_block = value ;}
  void set_X_block_norm (mat X_block) {
    for (arma::uword col_i = 0; col_i < X_block.n_cols; col_i ++) {
      if (arma::stddev(X_block.col(col_i)) == 0) {
        X_block.col(col_i) = (X_block.col(col_i) - arma::mean(X_block.col(col_i)));
      } else {
        X_block.col(col_i) = (X_block.col(col_i) - arma::mean(X_block.col(col_i))) / arma::stddev(X_block.col(col_i));
      }
    }
    m_X_block = X_block;
  }
  arma::mat get_X_block () { return m_X_block; }
  void set_y_mx (mat value) { m_y_mx = value ;}
  arma::mat get_y_mx () { return m_y_mx; }
  void set_sizes () {
    m_T = m_y_mx.n_rows;
    m_M = m_y_mx.n_cols;
    }
  void get_sizes () {
    std::cout << m_T << " " << m_M << "\n";
  }
  double get_v_0 () { return m_v_0; }
  arma::mat get_S_0 () { return m_S_0; }

  // approximate Dirac filter: populate m_delta
  void set_delta_mx (arma::umat given_delta_mx) {
    m_delta = given_delta_mx - 1;
    }
  void filter (const int &K) {
    m_delta.set_size(K, m_y_mx.n_cols);
    vec pip(m_X_block.n_cols, fill::zeros);
    for (uword i = 0; i < m_y_mx.n_cols; i++) {
      double yTy = dot(m_y_mx.col(i), m_y_mx.col(i));
      uvec order_v;
        for (uword j = 0; j < m_X_block.n_cols; j++) {
          pip(j) = calculate_pip(i, j, yTy);
        }
      order_v = sort_index(pip, "descent");
      m_delta.col(i) = arma::sort(order_v.head(K)); // sorting
    }
  }
  void sequential_filter (const int &K) {
    m_delta.set_size(K, m_y_mx.n_cols);
    for (uword i = 0; i < m_y_mx.n_cols; i++) { // for each series
      double yTy = dot(m_y_mx.col(i), m_y_mx.col(i));
      arma::uvec delta_k(K);
      arma::uvec possible_index_v = arma::regspace<arma::uvec>(0, m_X_block.n_cols - 1);
      for (arma::uword k = 0; k < K; k++) { // sequence over the number of predictors to be filtered K
        arma::vec pip(possible_index_v.n_elem);
        arma::uvec active_v, mask_possible;
        active_v = delta_k.head(k + 1);
        for (uword j = 0; j < possible_index_v.n_elem; j++) { // for each possible predictor
          active_v(k) = possible_index_v(j);
          pip(j) = calculate_sequential_pip(i, active_v, yTy);
        }
        delta_k(k) = possible_index_v(pip.index_max());
        mask_possible = possible_index_v != possible_index_v(pip.index_max());
        possible_index_v = possible_index_v(arma::find(mask_possible));
      }
      m_delta.col(i) = arma::sort(delta_k);
    }
  }
  umat get_delta_mx () { return m_delta; }

  // create block diagional covariate matrix X_alpha
  void fill_X_alpha () {
    m_X_alpha.set_size(m_X_block.n_rows * m_y_mx.n_cols, m_delta.n_rows * m_y_mx.n_cols);
    for (uword i = 0; i < m_y_mx.n_cols; i++) {
      m_X_alpha.submat(
        i * m_X_block.n_rows, i * m_delta.n_rows,
        (i + 1) * m_X_block.n_rows - 1, (i + 1) * m_delta.n_rows - 1
      ) = m_X_block.cols(m_delta.col(i));
    }
  }
  mat get_X_alpha () { return mat(m_X_alpha); }

  // create block diagional matrix of predictions

  // calculate optimal number of blocks
  double good_blocks () {
    double n = m_delta.n_elem;
    double big_term = pow(9 * n + sqrt(3 * (-1 + 27 * pow(n, 2))), .333333);
    double opt_B = 1 / (pow(3, .333333) * big_term) + big_term / pow(3, .666666);
    return floor(opt_B);
  }

  // SAMPLER
  void sample(const bool &RATS, const double &blocks,
    const int &iterations, const int &burn, const int &step_ahead,
    const bool &post_par, const bool &post_pred, const bool &update_w
  ) {
    ProgressBar pbar(iterations, 70);
    pbar.display();
    // helpers
    mat Kron_hlpr(m_X_block.n_rows, m_X_block.n_rows, fill::eye);
    vec y_vec = vectorise(m_y_mx);
    // hyperparameters Wishart
    set_Wishart_hp(RATS);
    // initial draws
    set_initial_draws();
    if (post_par) {
      initialize_lists(iterations, burn);
    } else {
      initialize_avg();
    }
    if (post_pred) initialize_post_sample(iterations, burn, step_ahead);
    GRR_update();
    compute_chol_inv_Sigma();
    for (size_t iter = 0; iter < iterations; iter++) {
      beta_update(Kron_hlpr, y_vec, blocks);
      kappa_update();
      inv_tau_sq_update();
      GRR_update();
      if (update_w) w_update();
      inv_Sigma_update(y_vec);
      compute_chol_inv_Sigma();
      if (iter >= burn) {
        compute_Sigma();
        if (post_par) {
          update_lists(iter, burn);
        } else {
          update_avg();
        }
        if (post_pred) update_post_sample(iter, burn, step_ahead);
      }
      ++pbar;
      pbar.display();
    }
    if (!post_par) normalize_avg(iterations, burn);
    pbar.done();
  }

  List show_estimates (const bool &post_par, const bool &post_pred) {
    List out;
    if (!post_par && !post_pred) { // only point estimates of pars
      out = List::create(
        _["delta.mx"] = m_delta + 1,
        _["beta"] = m_beta_avg,
        _["kappa"] = m_kappa_avg,
        _["tau.sq"] = 1 / m_inv_tau_sq_avg,
        _["omega"] = m_w_avg,
        _["Sigma"] = m_Sigma_avg,
        _["X.block"] = m_X_block
      );
    }
    if (post_par && !post_pred) { // only post distributions of pars
      out = List::create(
        _["delta.mx"] = m_delta + 1,
        _["beta"] = m_beta_list,
        _["kappa"] = m_kappa_list,
        _["tau.sq"] = m_tau_sq_list,
        _["omega"] = m_w_list,
        _["Sigma"] = m_Sigma_list,
        _["X.block"] = m_X_block
      );
    }
    if (!post_par && post_pred) { // point estimates of pars and posterior predictive
      out = List::create(
        _["delta.mx"] = m_delta + 1,
        _["beta"] = m_beta_avg,
        _["kappa"] = m_kappa_avg,
        _["tau.sq"] = 1 / m_inv_tau_sq_avg,
        _["omega"] = m_w_avg,
        _["Sigma"] = m_Sigma_avg,
        _["posterior.sample"] = m_post_sample,
        _["X.block"] = m_X_block
      );
    }
    if(post_par && post_pred) { // posterior of pars and predictive
      out = List::create(
        _["delta.mx"] = m_delta + 1,
        _["beta"] = m_beta_list,
        _["kappa"] = m_kappa_list,
        _["tau.sq"] = m_tau_sq_list,
        _["omega"] = m_w_list,
        _["Sigma"] = m_Sigma_list,
        _["posterior.sample"] = m_post_sample,
        _["X.block"] = m_X_block
      );
    }
    return out;
  }
};

/*
END OF JBU MODEL CLASS
START R FUNCTIONS
*/

//' Normalize a predictor block
//'
//' Centers and scales each column of `X_block`. Columns with non-zero
//' standard deviation are divided by their standard deviation. Columns with
//' zero standard deviation are centered only.
//'
//' @param X_block A numeric matrix of predictors.
//'
//' @return A numeric matrix with the same dimensions as `X_block`, where each
//' column has been centered and, when possible, scaled.
//'
//' @examples
//' X_block <- matrix(1:12, nrow = 4)
//' JBU_norm_X_block(X_block)
//'
//' @export
// [[Rcpp::export]]
arma::mat JBU_norm_X_block(arma::mat X_block) {
  JBU_model mod;
  mod.set_X_block_norm(X_block);
  return mod.get_X_block();
}

//' Select predictors for each series
//'
//' Applies the JBU filtering step to select the `K` most relevant predictors
//' from `X_block` for each series in `y_mx`.
//'
//' @param X_block A numeric matrix of candidate predictors.
//' @param y_mx A numeric matrix of observed series. Rows are time points and
//' columns are series.
//' @param K An integer giving the number of predictors to select for each series.
//'
//' @return An integer matrix with `K` rows and one column for each series in
//' `y_mx`. Each column contains the selected predictor indices.
//'
//' @examples
//' set.seed(1)
//' X_block <- matrix(rnorm(40), nrow = 10, ncol = 4)
//' y_mx <- cbind(
//'   2 * X_block[, 1] + rnorm(10, sd = 0.1),
//'   -1 * X_block[, 3] + rnorm(10, sd = 0.1)
//' )
//'
//' JBU_filter(X_block, y_mx, K = 2)
//'
//' @export
// [[Rcpp::export]]
arma::umat JBU_filter (const arma::mat &X_block, const arma::mat &y_mx, const int &K) {
  JBU_model mod;
  mod.set_X_block_norm(X_block);
  mod.set_y_mx(y_mx);
  mod.filter(K);
  return mod.get_delta_mx() + 1;
}

//' Build the filtered predictor matrix
//'
//' Applies the JBU filter and constructs the block-diagonal predictor matrix
//' used by the sampler.
//'
//' @param X_block A numeric matrix of candidate predictors.
//' @param y_mx A numeric matrix of observed series. Rows are time points and
//' columns are series.
//' @param K An integer giving the number of predictors to select for each series.
//'
//' @return A numeric matrix containing the filtered block-diagonal predictor
//' structure. If `X_block` has `T` rows, `y_mx` has `M` columns, and `K`
//' predictors are selected per series, the result has `T * M` rows and
//' `K * M` columns.
//'
//' @examples
//' set.seed(1)
//' X_block <- matrix(rnorm(40), nrow = 10, ncol = 4)
//' y_mx <- cbind(
//'   2 * X_block[, 1] + rnorm(10, sd = 0.1),
//'   -1 * X_block[, 3] + rnorm(10, sd = 0.1)
//' )
//'
//' X_alpha <- JBU_X_alpha(X_block, y_mx, K = 2)
//' dim(X_alpha)
//' X_alpha
//'
//' @export
// [[Rcpp::export]]
arma::mat JBU_X_alpha(const arma::mat &X_block, const arma::mat &y_mx, const int &K) {
  JBU_model mod;
  mod.set_X_block_norm(X_block);
  mod.set_y_mx(y_mx);
  mod.filter(K);
  mod.fill_X_alpha();
  return mod.get_X_alpha();
}

//' @export
// [[Rcpp::export]]
Rcpp::List JBU_sample(const arma::mat &X_block, const arma::mat &y_mx, const int &K,
  const bool RATS = true, double blocks = 1, const bool suggest_blocks = false,
  const int iterations = 1000, const int burn = 200, const int step_ahead = 4,
  const bool post_par = true, const bool post_pred = true, const bool update_w = true,
  const bool use_given_delta = false, Rcpp::Nullable<Rcpp::NumericMatrix> given_delta_in = R_NilValue,
  const double a_1 = 5, const double a_2 = 50, const bool do_sequential_filter = false
) {
  Rcpp::List out;
  JBU_model mod;
  mod.set_slab_parameters(a_1, a_2);
  mod.set_X_block_norm(X_block);
  mod.set_y_mx(y_mx);
  mod.set_sizes();
  if (use_given_delta & given_delta_in.isNotNull()) {
    arma::umat given_delta = as<umat>(given_delta_in);
    mod.set_delta_mx(given_delta);
  } else {
    if (do_sequential_filter) {
      mod.sequential_filter(K);
    } else {
      mod.filter(K);
    }
  }
  mod.fill_X_alpha();
  if (suggest_blocks) {
    blocks = mod.good_blocks();
    std::cout << "using " << blocks << " block(s).\n";
}
  mod.sample(RATS, blocks, iterations, burn, step_ahead, post_par, post_pred, update_w);
  out = mod.show_estimates(post_par, post_pred);
  return out;
}
