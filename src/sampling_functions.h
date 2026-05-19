/*
R SAMPLING DRAWING FUNCTIONS
INSTRUMENTAL TO JBU MODEL
COMPATIBLE WITH ARMADILLO
*/

#ifndef __SAMPLING_FUNCTIONS__
#define __SAMPLING_FUNCTIONS__


#include <RcppArmadillo.h>


arma::vec r_binom(const int &n, const arma::vec &p) {
  Rcpp::Function r_binom("rbinom"); //use this function from R in C
  Rcpp::NumericVector tmp = r_binom(Rcpp::_["n"] = n, Rcpp::_["size"] = 1, Rcpp::_["prob"] = p);
  arma::vec out(p.n_elem);
  for (arma::uword i = 0; i < p.n_elem; i++) {
    out(i) = tmp[i];
  }
  return out;
}
// why create a vector out just to save what is in tmp ?? 
// wouldn't it work if we create tmp with the right size p.n_elem from the beginning or if we reduce its size after ??
// also from the paper, can't p.n_elem > n and thus the for would not work ??


arma::vec r_gamma(const int &n, const double &shape, const arma::vec &rate) {
  Rcpp::Function r_gamma("rgamma");
  Rcpp::NumericVector tmp = r_gamma(Rcpp::_["n"] = n, Rcpp::_["shape"] = shape, Rcpp::_["rate"] = rate);
  arma::vec out(rate.n_elem);
  for (arma::uword i = 0; i < rate.n_elem; i++) {
    out(i) = tmp[i];
  }
  return out;
}
// same questions


double r_beta(const double &v, const double &V) {
  Rcpp::Function r_beta("rbeta");
  Rcpp::NumericVector out = r_beta(1, Rcpp::_["shape1"] = v, Rcpp::_["shape2"] = V);
  return out[0];
}

#endif
