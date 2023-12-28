// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"
#include <time.h>

#include <Eigen/Dense>

using namespace libgp;

int main (int argc, char const *argv[])
{
  int n=300, m=1000;     // n为训练数据数量，m为测试集数量
  double tss = 0, error, f, y;   // f为GP预测，y为实际数值，tss为误差累计，error为单次误差
  // initialize Gaussian process for 2-D input using the squared exponential 
  // covariance function with additive white noise.
  GaussianProcess gp(2, "CovSum ( CovSEiso, CovNoise)");    // 输入维度为2， 高斯核函数，白噪声
  // initialize hyper parameter vector
  Eigen::VectorXd params(gp.covf().get_param_dim());   // params_dim, ell, s_f, s_n
  std::cout << "param_dim = " << gp.covf().get_param_dim() <<  " input_dim = " << gp.covf().get_input_dim() <<std::endl;
  params << 0.0, 0.0, -2.0;    // ell, s_f, s_n
  // set parameters of covariance function
  gp.covf().set_loghyper(params);
  // add training patterns
  for(int i = 0; i < n; ++i) {
    double x[] = {drand48()*4-2, drand48()*4-2};    
    y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
    clock_t start = clock(); 
    gp.add_pattern(x, y);       // x为数组，y为double， 加入一组数据
    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "duration = " << duration << std::endl;
  }
  // total squared error
  for(int i = 0; i < m; ++i) {
    double x[] = {drand48()*4-2, drand48()*4-2};    // 实际当前状态数组，套入gp，得到f
    f = gp.f(x);
    y = Utils::hill(x[0], x[1]);
    error = f - y;
    tss += error*error;
  }
  std::cout << "mse = " << tss/m << std::endl;

  for (int i = 0; i < 300; ++i) {
    double x[] = {drand48()*4-2, drand48()*4-2, 10, 5, 1, 5};
    y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
    clock_t start = clock(); 
    gp.replace_pattern(x, y);
    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "duration_replace = " << duration << std::endl;
  }
  tss = 0.0;
  for(int i = 0; i < m; ++i) {
    double x[] = {drand48()*4-2, drand48()*4-2};    // 实际当前状态数组，套入gp，得到f
    clock_t start = clock(); 
    f = gp.f(x);
    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "duration f = " << duration << std::endl;
    y = Utils::hill(x[0], x[1]);
    error = f - y;
    tss += error*error;
  }
  std::cout << "mse final = " << tss/m << std::endl;
	return EXIT_SUCCESS;
}
