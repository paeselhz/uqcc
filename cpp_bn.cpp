// cpp_bn.cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


int sum_logical(LogicalVector v)
{
  int sum = 0;
  for (int i = 0; i < v.length(); i++)
  {
    if (v[i])
    {
      sum++;
    }
  }
  return (sum);
}


int sum_integer(IntegerVector v)
{
  int sum = 0;
  for (int idx = 0; idx < v.length(); idx++)
  {
    sum += v[idx];
  }
  return (sum);
}


int vector_min(IntegerVector v)
{
  
  IntegerVector::iterator it = std::min_element(v.begin(), v.end());
  
  return *it;
}


IntegerVector vector_positions(LogicalVector v)
{
  
  IntegerVector v_ret;
  for (int i = 0; i < v.length(); i++)
  {
    if (v[i])
    {
      v_ret.push_back(i);
    }
  }
  return v_ret;
}


double sum_matrix_groups(NumericMatrix md,
                         LogicalVector group1,
                         LogicalVector group2)
{
  arma::uvec vec_g1 = as<arma::uvec>(vector_positions(group1));
  arma::uvec vec_g2 = as<arma::uvec>(vector_positions(group2));
  arma::Mat<double> x = as<arma::mat>(md);
  arma::Mat<double> ret_mat = x.submat(vec_g1, vec_g2);
  
  double return_value = arma::accu(ret_mat);
  return return_value;
}

//[[Rcpp::export]]
double cpp_bn(NumericVector group_id, NumericMatrix md)
{
  double sBn;
  
  LogicalVector group1 = group_id == 0;
  LogicalVector group2 = group_id == 1;
  
  IntegerVector ngv = {sum_logical(group1),
                       sum_logical(group2)};
  
  double ng = md.nrow();
  
  if (ng != sum_integer(ngv))
  {
    stop("Error: Incorrect matrix dimension or group_id");
  }
  
  if (vector_min(ngv) > 1)
  {
    double s11 = sum_matrix_groups(md, group1, group1) / 2;
    double s22 = sum_matrix_groups(md, group2, group2) / 2;
    double s12 = sum_matrix_groups(md, group1, group2);
    
    double a1 = (1 / (ngv[0] * ngv[1])) * s12;
    double a2 = (2 / (ngv[0] * (ngv[0] - 1))) * s11;
    double a3 = (2 / (ngv[1] * (ngv[1] - 1))) * s22;
    sBn = (ngv[0] * ngv[1] / (ng * (ng - 1))) * (2 * a1 - a2 - a3);
  }
  else
  {
    if (ngv[0] == 1)
    {
      double s22 = sum_matrix_groups(md, group2, group2) / 2;
      double s12 = sum_matrix_groups(md, group1, group2);
      
      double a1 = (1 / ngv[0] * ngv[1]) * s12;
      double a2 = 0;
      double a3 = (2 / (ngv[1] * (ngv[1] - 1))) * s22;
      sBn = (ngv[0] * ngv[1] / (ng * (ng - 1))) * (a1 - a2 - a3);
    }
    else
    {
      double s11 = sum_matrix_groups(md, group1, group1) / 2.0;
      double s12 = sum_matrix_groups(md, group1, group2);
      
      double a1 = (1.0 / (ngv[0] * ngv[1])) * s12;
      double a2 = (2.0 / (ngv[0] * (ngv[0] - 1))) * s11;
      double a3 = 0.0;
      sBn = ((ngv[0] * ngv[1]) / (ng * (ng - 1))) * (a1 - a2 - a3);
    }
  }
  return sBn;
}