// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*! 
 *  
 *   \page licence Licensing
 *    
 *     libgp - Gaussian process library for Machine Learning
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __GP_H__
#define __GP_H__

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "cov.h"
#include "sampleset.h"
#include "vector"

namespace libgp {
  
  /** Gaussian process regression.
   *  @author Manuel Blum */
  class GaussianProcess
  {
  public:

    /** Empty initialization */
    GaussianProcess ();
    
    /** Create and instance of GaussianProcess with given input dimensionality 
     *  and covariance function. */
    GaussianProcess (size_t input_dim, std::string covf_def);
    
    /** Create and instance of GaussianProcess from file. */
    GaussianProcess (const char * filename);
    
    /** Copy constructor */
    GaussianProcess (const GaussianProcess& gp);
    
    virtual ~GaussianProcess ();
    
    /** Write current gp model to file. */
    void write(const char * filename);
    
    /** Predict target value for given input.
     *  @param x input vector
     *  @return predicted value */
    virtual double f(const double x[]);
    
    /** Predict variance of prediction for given input.
     *  @param x input vector
     *  @return predicted variance */
    virtual double var(const double x[]);
    
    /** Add input-output-pair to sample set.
     *  Add a copy of the given input-output-pair to sample set.
     *  @param x input array
     *  @param y output value
     */
    void add_pattern(const double x[], double y);
    void add_pattern(const double x[], double y, int t_now);


    /** replace unused data from sample set.
     *  @param x input array
     *  @param y output value
    */
    void replace_pattern(const double x[], double y, int t_now);
    void replace_pattern(const double x[], double y);
    void replace_pattern_2(const int i, const double x[], double y);


    bool set_y(size_t i, double y);

    /** Get number of samples in the training set. */
    size_t get_sampleset_size();
    
    /** Clear sample set and free memory. */
    void clear_sampleset();
    
    /** Get reference on currently used covariance function. */
    CovarianceFunction & covf();
    
    /** Get input vector dimensionality. */
    size_t get_input_dim();

    double log_likelihood();
    
    Eigen::VectorXd log_likelihood_gradient();

    bool need_add_pattern = false;

    std::vector<Eigen::MatrixXd> L_copy_lst;
    std::vector<Eigen::MatrixXd> L_lst;
    std::vector<Eigen::VectorXd> star_lst;
    std::vector<double> gamma_lst;
    std::vector<double> t_lst;
    double average_gamma;
    double sum_gamma = 0.0;
  protected:
    
    /** The covariance function of this Gaussian process. */
    CovarianceFunction * cf;
    
    /** The training sample set. */
    SampleSet * sampleset;
    
    /** Alpha is cached for performance. */ 
    Eigen::VectorXd alpha;
    Eigen::VectorXd gamma_star;
    Eigen::MatrixXd L_hist;
    Eigen::MatrixXd L_copy_hist;
    Eigen::VectorXd star_hist;
    
    /** Last test kernel vector. */
    Eigen::VectorXd k_star;

    /** Linear solver used to invert the covariance matrix. */
//    Eigen::LLT<Eigen::MatrixXd> solver;
    Eigen::MatrixXd L;
    Eigen::MatrixXd L_copy;
    
    /** Input vector dimensionality. */
    size_t input_dim;
    
    /** Update test input and cache kernel vector. */
    void update_k_star(const Eigen::VectorXd &x_star);

    void update_alpha();

    /** Compute covariance matrix and perform cholesky decomposition. */
    virtual void compute();
    
    bool alpha_needs_update;

    double cal_gamma(Eigen::MatrixXd L_now, const Eigen::VectorXd &x_input, int index);
    double cal_gamma_replace(Eigen::MatrixXd L_now, const Eigen::VectorXd &x_input, int min_i);


  private:

    /** No assignement */
    GaussianProcess& operator=(const GaussianProcess&);

  };
}

#endif /* __GP_H__ */
