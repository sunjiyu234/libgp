// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"
#include <time.h>

#include <Eigen/Dense>
#include "CSVReader.h"
#include <vector>
#include <string.h>
#include "cg.h"

using namespace libgp;
using namespace std;

int main (int argc, char const *argv[])
{
  int n = 300;
  int  m=1000;     // n为训练数据数量，m为测试集数量
  double tss_v = 0, tss_r = 0, tss_beta = 0, error_v, f_v, y_v, error_r, f_r, y_r, error_beta, f_beta, y_beta;   // f为GP预测，y为实际数值，tss为误差累计，error为单次误差
  // initialize Gaussian process for 2-D input using the squared exponential 
  // covariance function with additive white noise.
  GaussianProcess *gp_v = new GaussianProcess(6, "CovSum ( CovSEiso, CovNoise)");    // 输入维度为6， 高斯核函数，白噪声
  GaussianProcess *gp_r = new GaussianProcess(6, "CovSum ( CovSEiso, CovNoise)");    // 输入维度为6， 高斯核函数，白噪声
  GaussianProcess *gp_beta = new GaussianProcess(6, "CovSum ( CovSEiso, CovNoise)");    // 输入维度为6， 高斯核函数，白噪声
  ofstream data_record_file_;
  data_record_file_.open("/home/sun234/record_gp_with_time_different_param_2.csv");
  vector<double> v, r, beta, delta, Tf, Tr;
  vector<double> v_error, r_error, beta_error;
  string data_file_v = "/home/sun234/1222/v.txt";
  string data_file_r = "/home/sun234/1222/r.txt";
  string data_file_beta = "/home/sun234/1222/beta.txt";
  string data_file_delta = "/home/sun234/1222/Steer_pre.txt";
  string data_file_Tf = "/home/sun234/1222/Tf_pre.txt";
  string data_file_Tr = "/home/sun234/1222/Tr_pre.txt";
  string data_file_v_error = "/home/sun234/1222/v_error.txt";
  string data_file_r_error = "/home/sun234/1222/r_error.txt";
  string data_file_beta_error = "/home/sun234/1222/beta_error.txt";
  CSVReader reader_v(data_file_v);
  vector<vector<string>> data_list_v = reader_v.getData();
  CSVReader reader_r(data_file_r);
  vector<vector<string>> data_list_r = reader_r.getData();
  CSVReader reader_beta(data_file_beta);
  vector<vector<string>> data_list_beta = reader_beta.getData();
  CSVReader reader_delta(data_file_delta);
  vector<vector<string>> data_list_delta = reader_delta.getData();
  CSVReader reader_Tf(data_file_Tf);
  vector<vector<string>> data_list_Tf = reader_Tf.getData();
  CSVReader reader_Tr(data_file_Tr);
  vector<vector<string>> data_list_Tr = reader_Tr.getData();
  CSVReader reader_v_error(data_file_v_error);
  vector<vector<string>> data_list_v_error = reader_v_error.getData();
  CSVReader reader_r_error(data_file_r_error);
  vector<vector<string>> data_list_r_error = reader_r_error.getData();
  CSVReader reader_beta_error(data_file_beta_error);
  vector<vector<string>> data_list_beta_error = reader_beta_error.getData();
  for(size_t i = 0; i < data_list_v.size() - 2; i++){
    v.push_back(stof(data_list_v.at(i).at(0)));
    r.push_back(stof(data_list_r.at(i).at(0)));
    beta.push_back(stof(data_list_beta.at(i).at(0)));
    delta.push_back(stof(data_list_delta.at(i).at(0)));
    Tf.push_back(stof(data_list_Tf.at(i).at(0)));
    Tr.push_back(stof(data_list_Tr.at(i).at(0)));
    v_error.push_back(stof(data_list_v_error.at(i).at(0)));
    r_error.push_back(stof(data_list_r_error.at(i).at(0)));
    beta_error.push_back(stof(data_list_beta_error.at(i).at(0)));
  }
  // initialize hyper parameter vector
  int num_total = v.size();
  Eigen::VectorXd params_v(gp_v->covf().get_param_dim());   // params_dim, ell, s_f, s_n
  Eigen::VectorXd params_r(gp_r->covf().get_param_dim());   // params_dim, ell, s_f, s_n
  Eigen::VectorXd params_beta(gp_beta->covf().get_param_dim());   // params_dim, ell, s_f, s_n
  std::cout << "param_dim = " << gp_v->covf().get_param_dim() <<  " input_dim = " << gp_v->covf().get_input_dim() <<std::endl;
  // params_v << 6.67699, 2.90749, 3.05803;
  // params_r << 6.16641, 7.67367, 3.2702;
  // params_beta << 8.00548, 8.86069, 2.83372;    // ell, s_f, s_n
  params_v << 8.27716, 6.25741, 3.72275;
  params_r << 5.81545, 6.06309, 3.12376;
  params_beta << 5.74294, 5.56467, 2.38394;    // ell, s_f, s_n
  // set parameters of covariance function
  gp_v->covf().set_loghyper(params_v);
  gp_r->covf().set_loghyper(params_r);
  gp_beta->covf().set_loghyper(params_beta);
    // total squared error
  int num_ok = 0;
  for(int i = 0; i < num_total; i++){
    cout << " i = " << i << endl;
    double x[] = {v.at(i), r.at(i) * 100.0, beta.at(i) * 1000.0, delta.at(i) * 1000.0, Tf.at(i), Tr.at(i)};
    double x_v[] = {v.at(i) * 10.0, r.at(i) * 10.0, beta.at(i) * 100.0, delta.at(i) * 100.0, Tf.at(i) * 10.0, Tr.at(i) * 10.0};

    f_v = gp_v->f(x_v) / 10000.0;
    double var_v = gp_v->var(x_v);
    y_v = v_error.at(i);
    cout << "f_v = " << f_v << " y_v = "<< y_v <<  " var_v = " << var_v << endl;
    error_v = f_v - y_v;
    tss_v += abs(error_v);

    f_r = gp_r->f(x) / 1000.0;
    double var_r = gp_r->var(x);
    y_r = r_error.at(i);
    error_r = f_r - y_r;
    cout << "f_r = " << f_r << " y_r = "<< y_r << " var_r = " << var_r << endl;
    tss_r += abs(error_r);

    f_beta = gp_beta->f(x) / 10000.0;
    double var_beta = gp_beta->var(x);
    y_beta = beta_error.at(i);
    error_beta = f_beta - y_beta;
    cout << "f_beta = " << f_beta << " y_beta = "<< y_beta << " var_beta = " << var_beta << endl;
    tss_beta += abs(error_beta);

    data_record_file_ << f_v << ","<< y_v << "," << var_v << "," << error_v << "," << f_r << ","<< y_r << "," << var_r << "," << error_r << "," <<  f_beta << ","<< y_beta << "," << var_beta << "," << error_beta << endl;

    if (abs(error_beta / y_beta) < 0.5 &&  abs(error_v / y_v) < 0.5 && abs(error_r / y_r) < 0.5){
      num_ok++;
    }

    if (i < n){
      gp_v->add_pattern(x_v, v_error[i] * 10000.0);
      gp_r->add_pattern(x, r_error[i] * 1000.0);
      gp_beta->add_pattern(x, beta_error[i] * 10000.0);
    }
    else{
      gp_v->replace_pattern(x_v, v_error[i]* 10000.0);
      gp_r->replace_pattern(x, r_error[i] * 1000.0);
      gp_beta->replace_pattern(x, beta_error[i]* 10000.0);
    }
  }

  std::cout << "mse_v = " << tss_v/num_total << std::endl;
  std::cout << "mse_r = " << tss_r/num_total << std::endl;
  std::cout << "mse_beta = " << tss_beta/num_total << std::endl;
  std::cout << "num_ok = " << num_ok << std::endl; 
  data_record_file_.close();

  delete gp_r;
  delete gp_v;
  delete gp_beta;
	return EXIT_SUCCESS;
}
