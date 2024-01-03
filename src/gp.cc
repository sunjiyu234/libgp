// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "cov_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {
  
  const double log2pi = log(2*M_PI);
  const double initial_L_size = 1000;

  GaussianProcess::GaussianProcess ()
  {
      sampleset = NULL;
      cf = NULL;
  }

  GaussianProcess::GaussianProcess (size_t input_dim, std::string covf_def)
  {
    // set input dimensionality
    this->input_dim = input_dim;
    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    cf->loghyper_changed = 0;
    sampleset = new SampleSet(input_dim);
    L.resize(initial_L_size, initial_L_size);
    L_copy.resize(initial_L_size, initial_L_size);
  }
  
  GaussianProcess::GaussianProcess (const char * filename) 
  {
    sampleset = NULL;
    cf = NULL;
    int stage = 0;
    std::ifstream infile;
    double y;
    infile.open(filename);
    std::string s;
    double * x = NULL;
    L.resize(initial_L_size, initial_L_size);
    while (infile.good()) {
      getline(infile, s);
      // ignore empty lines and comments
      if (s.length() != 0 && s.at(0) != '#') {
        std::stringstream ss(s);
        if (stage > 2) {
          ss >> y;
          for(size_t j = 0; j < input_dim; ++j) {
            ss >> x[j];
          }
          add_pattern(x, y);
        } else if (stage == 0) {
          ss >> input_dim;
          sampleset = new SampleSet(input_dim);
          x = new double[input_dim];
        } else if (stage == 1) {
          CovFactory factory;
          cf = factory.create(input_dim, s);
          cf->loghyper_changed = 0;
        } else if (stage == 2) {
          Eigen::VectorXd params(cf->get_param_dim());
          for (size_t j = 0; j<cf->get_param_dim(); ++j) {
            ss >> params[j];
          }
          cf->set_loghyper(params);
        }
        stage++;
      }
    }
    infile.close();
    if (stage < 3) {
      std::cerr << "fatal error while reading " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    delete [] x;
  }
  
  GaussianProcess::GaussianProcess(const GaussianProcess& gp)
  {
    this->input_dim = gp.input_dim;
    sampleset = new SampleSet(*(gp.sampleset));
    alpha = gp.alpha;
    k_star = gp.k_star;
    alpha_needs_update = gp.alpha_needs_update;
    L = gp.L;
    
    // copy covariance function
    CovFactory factory;
    cf = factory.create(gp.input_dim, gp.cf->to_string());
    cf->loghyper_changed = gp.cf->loghyper_changed;
    cf->set_loghyper(gp.cf->get_loghyper());
  }
  
  GaussianProcess::~GaussianProcess ()
  {
    // free memory
    if (sampleset != NULL) delete sampleset;
    if (cf != NULL) delete cf;
  }  
  
  double GaussianProcess::f(const double x[])
  {
    if (sampleset->empty()) return 0;
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    compute();
    update_alpha();
    update_k_star(x_star);
    return k_star.dot(alpha);    
  }
  
  double GaussianProcess::var(const double x[])
  {
    if (sampleset->empty()) return 0;
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    compute();
    update_alpha();
    update_k_star(x_star);
    int n = sampleset->size();
    Eigen::VectorXd v_gp = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star);
    return cf->get(x_star, x_star) - v_gp.dot(v_gp);	
  }

  void GaussianProcess::compute()
  {
    // can previously computed values be used?
    if (!cf->loghyper_changed) return;
    cf->loghyper_changed = false;
    int n_size = sampleset->size();
    // resize L if necessary
    if (n_size > L.rows()) L.resize(n_size + initial_L_size, n_size + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i_compute = 0; i_compute < sampleset->size(); ++i_compute) {
      for(size_t j_compute = 0; j_compute <= i_compute; ++j_compute) {
        L(i_compute, j_compute) = cf->get(sampleset->x(i_compute), sampleset->x(j_compute));
      }
    }
    L_copy = L;
    std::cout << "UPDATE" << std::endl;
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    L.topLeftCorner(n_size, n_size) = L.topLeftCorner(n_size, n_size).selfadjointView<Eigen::Lower>().llt().matrixL();
    alpha_needs_update = true;
  }
  
  void GaussianProcess::update_k_star(const Eigen::VectorXd &x_star)
  {
    k_star.resize(sampleset->size());
    for(size_t i_k_star = 0; i_k_star < sampleset->size(); ++i_k_star) {
      k_star(i_k_star) = cf->get(x_star, sampleset->x(i_k_star));
    }
  }

  void GaussianProcess::update_alpha()
  {
    // can previously computed values be used?
    if (!alpha_needs_update) return;
    alpha_needs_update = false;
    alpha.resize(sampleset->size());
    // Map target values to VectorXd
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    int n_size = sampleset->size();
    alpha = L.topLeftCorner(n_size, n_size).triangularView<Eigen::Lower>().solve(y);    // cholesky分解
    std::cout << "before = " << alpha[0] << std::endl;
    L.topLeftCorner(n_size, n_size).triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha);     // alpha = (LTL)-1Y
    std::cout << "after = " << alpha[0] << std::endl;
  }
  
  void GaussianProcess::add_pattern(const double x[], double y)
  {
    clock_t start_add = clock(); 
    //std::cout<< L.rows() << std::endl;
#if 0
    sampleset->add(x, y);
    cf->loghyper_changed = true;
    alpha_needs_update = true;
    cached_x_star = NULL;
    return;
#else
    int n = sampleset->size();    
    sampleset->add(x, y);
    // create kernel matrix if sampleset is empty
    if (n == 0) {
      L(0,0) = sqrt(cf->get(sampleset->x(0), sampleset->x(0)));    // sampleset->x(0)为vector<double>
      cf->loghyper_changed = false;
      L_copy(0,0) = cf->get(sampleset->x(0), sampleset->x(0));
      alpha_needs_update = true;
      need_add_pattern = false;
      return;
    // recompute kernel matrix if necessary
    } else if (cf->loghyper_changed) {
      compute();
    // update kernel matrix 
    } else {
      Eigen::VectorXd k(n);
      for (int i = 0; i<n; ++i) {
        k(i) = cf->get(sampleset->x(i), sampleset->x(n));
      }
      double kappa = cf->get(sampleset->x(n), sampleset->x(n));
      // resize L if necessary
      if (sampleset->size() > static_cast<std::size_t>(L.rows())) {
        L.conservativeResize(n + initial_L_size, n + initial_L_size);
        L_copy.conservativeResize(n + initial_L_size, n + initial_L_size);
      }
      L_copy.block(n, 0, 1, n) = k.transpose();
      L_copy(n, n) = kappa;
      L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
      L.block(n,0,1,n) = k.transpose();
      L(n,n) = sqrt(kappa - k.dot(k));
    }

    if (n == 1){
      //  添加点后，点集中包含两个点，初始构建记忆函数gamma
      Eigen::MatrixXd L_copy_0(1, 1);
      L_copy_0(0, 0) = cf->get(sampleset->x(1), sampleset->x(1));
      Eigen::MatrixXd L_copy_1(1, 1);
      L_copy_1(0, 0) = L_copy(0, 0);
      L_copy_lst = {L_copy_0, L_copy_1};
      Eigen::MatrixXd L_0(1, 1);
      L_0.topLeftCorner(1, 1) = L_copy_0.topLeftCorner(1, 1).selfadjointView<Eigen::Lower>().llt().matrixL();
      Eigen::MatrixXd L_1(1, 1);
      L_1.topLeftCorner(1, 1) = L_copy_1.topLeftCorner(1, 1).selfadjointView<Eigen::Lower>().llt().matrixL();
      L_lst = {L_0, L_1};
      double gamma_0 = cal_gamma(L_0, sampleset->x(0), 1);
      star_lst.push_back(gamma_star);
      double gamma_1 = cal_gamma(L_1, sampleset->x(1), 0);
      star_lst.push_back(gamma_star);
      gamma_lst = {gamma_0, gamma_1};
      sum_gamma += (gamma_0 + gamma_1);
    }else{
      int total_size = L_lst.size();
      Eigen::VectorXd k_hist(total_size);
      Eigen::VectorXd k_hist_cir(total_size - 1);
      for (int index_x = 0; index_x < total_size; ++index_x){
        k_hist(index_x) = cf->get(sampleset->x(index_x), sampleset->x(n));
      }
      for (int index_hist = 0; index_hist < total_size; ++index_hist){
        clock_t start_add_2 = clock(); 
        star_hist = star_lst[index_hist];
        double kappa_hist = cf->get(sampleset->x(n), sampleset->x(n));
        k_hist_cir.head(index_hist) = k_hist.head(index_hist);
        k_hist_cir.tail(total_size - index_hist - 1) = k_hist.tail(total_size - index_hist - 1);
        if (sampleset->size() > static_cast<std::size_t>(L_lst[index_hist].rows())) {
          L_lst[index_hist].conservativeResize(n + initial_L_size, n + initial_L_size);
          L_copy_lst[index_hist].conservativeResize(n + initial_L_size, n + initial_L_size);
        }
        clock_t end_add_2 = clock(); 
        double duration_add_2 = (double)(end_add_2 - start_add_2) / CLOCKS_PER_SEC;
        // std::cout << "duration_add_2 = " << duration_add_2 << std::endl;
        L_copy_lst[index_hist].block(total_size - 1, 0, 1, total_size - 1) = k_hist_cir.transpose();
        L_copy_lst[index_hist](total_size - 1, total_size - 1) = kappa_hist;
        L_lst[index_hist].topLeftCorner(total_size - 1, total_size - 1).triangularView<Eigen::Lower>().solveInPlace(k_hist_cir);
        L_lst[index_hist].block(total_size - 1, 0, 1, total_size - 1) = k_hist_cir.transpose();
        L_lst[index_hist](total_size - 1, total_size - 1) = sqrt(kappa_hist - k_hist_cir.dot(k_hist_cir));
        star_hist.conservativeResize(total_size);
        star_hist(total_size - 1) = cf->get(sampleset->x(index_hist),sampleset->x(n)); 
        star_lst[index_hist] = star_hist;
        Eigen::VectorXd gamma_gp = L_lst[index_hist].topLeftCorner(total_size, total_size).triangularView<Eigen::Lower>().solve(star_hist);
        sum_gamma -= gamma_lst[index_hist];
        // std::cout << "gamma_before = " << gamma_lst[index_hist] << std::endl;
        gamma_lst[index_hist] = cf->get(sampleset->x(index_hist), sampleset->x(index_hist)) - gamma_gp.dot(gamma_gp);
        // std::cout << "gamma_after = " << gamma_lst[index_hist] << std::endl;
        sum_gamma += gamma_lst[index_hist];
      }


      Eigen::MatrixXd L_copy_i(n , n);
      L_copy_i = L_copy.topLeftCorner(n, n);
      L_copy_lst.push_back(L_copy_i);

      Eigen::MatrixXd L_i(n, n);
      L_i = L.topLeftCorner(n, n);
      L_lst.push_back(L_i);

      double gamma_i = cal_gamma(L_i, sampleset->x(n), 0);
      star_lst.push_back(gamma_star);
      std::cout << "gamma_i = " << gamma_i << std::endl;
      gamma_lst.push_back(gamma_i);
      sum_gamma += gamma_i;
      average_gamma = sum_gamma / gamma_lst.size();
      std::cout << "average_gamma = " << average_gamma << std::endl;
    }
    alpha_needs_update = true;
    need_add_pattern = false;
    clock_t end_add = clock();
    double duration_add = (double)(end_add - start_add) / CLOCKS_PER_SEC;
    std::cout << "duration_add = " << duration_add << std::endl;
#endif
  }

  void GaussianProcess::add_pattern(const double x[], double y, int t_now)
  {
    t_lst.push_back(t_now);
    clock_t start_add = clock(); 
    //std::cout<< L.rows() << std::endl;
#if 0
    sampleset->add(x, y);
    cf->loghyper_changed = true;
    alpha_needs_update = true;
    cached_x_star = NULL;
    return;
#else
    int n = sampleset->size();    
    sampleset->add(x, y);
    // create kernel matrix if sampleset is empty
    if (n == 0) {
      L(0,0) = sqrt(cf->get(sampleset->x(0), sampleset->x(0)));    // sampleset->x(0)为vector<double>
      cf->loghyper_changed = false;
      L_copy(0,0) = cf->get(sampleset->x(0), sampleset->x(0));
      alpha_needs_update = true;
      need_add_pattern = false;
      return;
    // recompute kernel matrix if necessary
    } else if (cf->loghyper_changed) {
      compute();
    // update kernel matrix 
    } else {
      Eigen::VectorXd k(n);
      for (int i = 0; i<n; ++i) {
        k(i) = cf->get(sampleset->x(i), sampleset->x(n));
      }
      double kappa = cf->get(sampleset->x(n), sampleset->x(n));
      // resize L if necessary
      if (sampleset->size() > static_cast<std::size_t>(L.rows())) {
        L.conservativeResize(n + initial_L_size, n + initial_L_size);
        L_copy.conservativeResize(n + initial_L_size, n + initial_L_size);
      }
      L_copy.block(n, 0, 1, n) = k.transpose();
      L_copy(n, n) = kappa;
      L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
      L.block(n,0,1,n) = k.transpose();
      L(n,n) = sqrt(kappa - k.dot(k));
    }

    if (n == 1){
      //  添加点后，点集中包含两个点，初始构建记忆函数gamma
      Eigen::MatrixXd L_copy_0(1, 1);
      L_copy_0(0, 0) = cf->get(sampleset->x(1), sampleset->x(1));
      Eigen::MatrixXd L_copy_1(1, 1);
      L_copy_1(0, 0) = L_copy(0, 0);
      L_copy_lst = {L_copy_0, L_copy_1};
      Eigen::MatrixXd L_0(1, 1);
      L_0.topLeftCorner(1, 1) = L_copy_0.topLeftCorner(1, 1).selfadjointView<Eigen::Lower>().llt().matrixL();
      Eigen::MatrixXd L_1(1, 1);
      L_1.topLeftCorner(1, 1) = L_copy_1.topLeftCorner(1, 1).selfadjointView<Eigen::Lower>().llt().matrixL();
      L_lst = {L_0, L_1};
      double gamma_0 = cal_gamma(L_0, sampleset->x(0), 1);
      star_lst.push_back(gamma_star);
      double gamma_1 = cal_gamma(L_1, sampleset->x(1), 0);
      star_lst.push_back(gamma_star);
      gamma_lst = {gamma_0, gamma_1};
      sum_gamma += (gamma_0 + gamma_1);
    }else{
      int total_size = L_lst.size();
      Eigen::VectorXd k_hist(total_size);
      Eigen::VectorXd k_hist_cir(total_size - 1);
      for (int index_x = 0; index_x < total_size; ++index_x){
        k_hist(index_x) = cf->get(sampleset->x(index_x), sampleset->x(n));
      }
      for (int index_hist = 0; index_hist < total_size; ++index_hist){
        clock_t start_add_2 = clock(); 
        star_hist = star_lst[index_hist];
        double kappa_hist = cf->get(sampleset->x(n), sampleset->x(n));
        k_hist_cir.head(index_hist) = k_hist.head(index_hist);
        k_hist_cir.tail(total_size - index_hist - 1) = k_hist.tail(total_size - index_hist - 1);
        if (sampleset->size() > static_cast<std::size_t>(L_lst[index_hist].rows())) {
          L_lst[index_hist].conservativeResize(n + initial_L_size, n + initial_L_size);
          L_copy_lst[index_hist].conservativeResize(n + initial_L_size, n + initial_L_size);
        }
        clock_t end_add_2 = clock(); 
        double duration_add_2 = (double)(end_add_2 - start_add_2) / CLOCKS_PER_SEC;
        // std::cout << "duration_add_2 = " << duration_add_2 << std::endl;
        L_copy_lst[index_hist].block(total_size - 1, 0, 1, total_size - 1) = k_hist_cir.transpose();
        L_copy_lst[index_hist](total_size - 1, total_size - 1) = kappa_hist;
        L_lst[index_hist].topLeftCorner(total_size - 1, total_size - 1).triangularView<Eigen::Lower>().solveInPlace(k_hist_cir);
        L_lst[index_hist].block(total_size - 1, 0, 1, total_size - 1) = k_hist_cir.transpose();
        L_lst[index_hist](total_size - 1, total_size - 1) = sqrt(kappa_hist - k_hist_cir.dot(k_hist_cir));
        star_hist.conservativeResize(total_size);
        star_hist(total_size - 1) = cf->get(sampleset->x(index_hist),sampleset->x(n)); 
        star_lst[index_hist] = star_hist;
        Eigen::VectorXd gamma_gp = L_lst[index_hist].topLeftCorner(total_size, total_size).triangularView<Eigen::Lower>().solve(star_hist);
        sum_gamma -= gamma_lst[index_hist];
        // std::cout << "gamma_before = " << gamma_lst[index_hist] << std::endl;
        gamma_lst[index_hist] = (cf->get(sampleset->x(index_hist), sampleset->x(index_hist)) - gamma_gp.dot(gamma_gp)) * exp(-(t_now - t_lst[index_hist]) *(t_now - t_lst[index_hist]) / (2.0 *  2500000.0));
        // std::cout << "gamma_after = " << gamma_lst[index_hist] << std::endl;
        sum_gamma += gamma_lst[index_hist];
      }


      Eigen::MatrixXd L_copy_i(n , n);
      L_copy_i = L_copy.topLeftCorner(n, n);
      L_copy_lst.push_back(L_copy_i);

      Eigen::MatrixXd L_i(n, n);
      L_i = L.topLeftCorner(n, n);
      L_lst.push_back(L_i);

      double gamma_i = cal_gamma(L_i, sampleset->x(n), 0);
      star_lst.push_back(gamma_star);
      std::cout << "gamma_i = " << gamma_i << std::endl;
      gamma_lst.push_back(gamma_i);
      sum_gamma += gamma_i;
      average_gamma = sum_gamma / gamma_lst.size();
      std::cout << "average_gamma = " << average_gamma << std::endl;
    }
    alpha_needs_update = true;
    need_add_pattern = false;
    clock_t end_add = clock();
    double duration_add = (double)(end_add - start_add) / CLOCKS_PER_SEC;
    std::cout << "duration_add = " << duration_add << std::endl;
#endif
  }

  void GaussianProcess::replace_pattern(const double x[], double y){
    clock_t start_replace = clock(); 
    std::vector<double>::iterator min_iter = std::min_element(gamma_lst.begin(), gamma_lst.end());
    int min_index = std::distance(gamma_lst.begin(), min_iter);
    // calculate gamma to decide add or not
    Eigen::Map<const Eigen::VectorXd> x_now(x, input_dim);
    std::cout << "min_index = " << min_index << std::endl;
    if(min_index > 1 && min_index < gamma_lst.size() - 1){
      std::cout << "min_index - 1 = " << gamma_lst[min_index - 1] << " min_index = " << gamma_lst[min_index] << " min_index + 1 = " << gamma_lst[min_index + 1] << std::endl;
    }
    double gamma_new = cal_gamma_replace(L_lst[min_index].topLeftCorner(sampleset->size() - 1, sampleset->size() - 1), x_now, min_index);
    star_lst[min_index] = gamma_star;
    std::cout << "gamma new= " << gamma_new << std::endl;
    if (gamma_new < average_gamma){
      std::cout << "gamma = " << gamma_new << "Useless Data\n";
      return;
    }
    sum_gamma -= gamma_lst[min_index];


    Eigen::VectorXd *v_rp = new Eigen::VectorXd(input_dim);
    for (size_t j=0; j<input_dim; ++j) (*v_rp)(j) = x[j];
    const Eigen::VectorXd *v2 = new Eigen::VectorXd(*v_rp);
    sampleset->erase_num(min_index);    // 删除第i个
    int n = sampleset->size();
    std::cout << "n = " << n << std::endl;
    Eigen::VectorXd k(n);
    for (int j = 0; j < n; ++j){
      k(j) = cf->get(sampleset->x(j), *v2);
    }
    double kappa = cf->get(*v2, *v2);
    if (cf->loghyper_changed){
      sampleset->insert_num(min_index, x, y);
      compute();
    } else{
      if(min_index > 0){
        L_copy.block(min_index, 0, 1, min_index) = k.head(min_index).transpose();
      }
      if (min_index < n){
        L_copy.block(min_index + 1, min_index, n - min_index, 1) = k.tail(n - min_index);
      }
      L_copy(min_index, min_index) = kappa;
      L.topLeftCorner(n+1, n+1) = L_copy.topLeftCorner(n+1, n+1).selfadjointView<Eigen::Lower>().llt().matrixL();
      sampleset->insert_num(min_index, x, y);
    }
    alpha_needs_update = true;
    delete v_rp;
    delete v2;

    int total_size = L_copy_lst.size();   // 共300个点
    // std::cout << "total_size = " << total_size << std::endl;
    Eigen::VectorXd k_hist_cir(total_size - 2);    // 除去index_hist点与被替换的点， 共298个点
    int num_cal  = 0;
    for(int index_hist = 0; index_hist < total_size; ++index_hist){ 
      if (index_hist == min_index){
        num_cal = 1;
        continue;
      }
      k_hist_cir.head(index_hist - num_cal) = k.head(index_hist - num_cal);     // k为299个点与新替换的点的核函数，k_hist_cir为k除去index_hist点，298个点
      k_hist_cir.tail(total_size - (index_hist - num_cal) - 2) = k.tail(total_size - (index_hist - num_cal) - 2);
      if (index_hist < min_index){
        if(min_index > 0){
          L_copy_lst[index_hist].block(min_index - 1, 0, 1, min_index - 1) = k_hist_cir.head(min_index - 1).transpose();
        }
        if (min_index < n){
          L_copy_lst[index_hist].block(min_index, min_index - 1, n - min_index, 1) = k_hist_cir.tail(n - min_index);
        }
        L_copy_lst[index_hist](min_index - 1, min_index - 1) = kappa;
        star_lst[index_hist](min_index - 1) = cf->get(sampleset->x(index_hist),sampleset->x(min_index));    // star中为index_hist与每一个其他的核函数；
      }else{
        if(min_index > 0){
          L_copy_lst[index_hist].block(min_index, 0, 1, min_index) = k_hist_cir.head(min_index).transpose();
        }
        if (min_index < n){
          L_copy_lst[index_hist].block(min_index + 1, min_index, n - min_index - 1, 1) = k_hist_cir.tail(n - min_index - 1);
        }
        L_copy_lst[index_hist](min_index, min_index) = kappa;
        star_lst[index_hist](min_index) = cf->get(sampleset->x(index_hist),sampleset->x(min_index));    // star中为index_hist与每一个其他的核函数；
      }
      L_lst[index_hist].topLeftCorner(n, n) = L_copy_lst[index_hist].topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
      star_hist = star_lst[index_hist];
      Eigen::VectorXd gamma_gp = L_lst[index_hist].topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(star_hist);
      sum_gamma -= gamma_lst[index_hist];
      // std::cout << "gamma_before = " << gamma_lst[index_hist] << std::endl;
      gamma_lst[index_hist] = (cf->get(sampleset->x(index_hist), sampleset->x(index_hist)) - gamma_gp.dot(gamma_gp));
      // std::cout << "index_hist = " << index_hist << std::endl;
      // std::cout << "gamma_1 = " << cf->get(sampleset->x(index_hist), sampleset->x(index_hist)) << " gamma_2 = " << gamma_gp.dot(gamma_gp) << std::endl;
      // std::cout << "gamma_after = " << gamma_lst[index_hist] << std::endl;
       if (gamma_lst[index_hist] < 0.0){
        int flag = getchar();
       }
      sum_gamma += gamma_lst[index_hist];
    }
    std::cout << "before_gamma = " << gamma_lst[min_index] << std::endl;
    gamma_lst[min_index] = gamma_new;
    std::cout << "after_gamma = " << gamma_lst[min_index] << std::endl;
    sum_gamma += gamma_lst[min_index];
    average_gamma = sum_gamma / gamma_lst.size();
    std::cout << "average_gamma = " << average_gamma << std::endl;
    clock_t end_replace = clock();
    double duration_replace = (double)(end_replace - start_replace) / CLOCKS_PER_SEC;
    std::cout << "duration_replace = " << duration_replace << std::endl;
  }


  void GaussianProcess::replace_pattern(const double x[], double y, int t_now){
    clock_t start_replace = clock(); 
    std::vector<double>::iterator min_iter = std::min_element(gamma_lst.begin(), gamma_lst.end());
    int min_index = std::distance(gamma_lst.begin(), min_iter);
    // calculate gamma to decide add or not
    Eigen::Map<const Eigen::VectorXd> x_now(x, input_dim);
    std::cout << "min_index = " << min_index << std::endl;
    double gamma_new = cal_gamma_replace(L_lst[min_index].topLeftCorner(sampleset->size() - 1, sampleset->size() - 1), x_now, min_index);
    star_lst[min_index] = gamma_star;
    std::cout << "gamma new= " << gamma_new << std::endl;
    if (gamma_new < average_gamma){
      std::cout << "gamma = " << gamma_new << "Useless Data\n";
      return;
    }
    sum_gamma -= gamma_lst[min_index];


    Eigen::VectorXd *v_rp = new Eigen::VectorXd(input_dim);
    for (size_t j=0; j<input_dim; ++j) (*v_rp)(j) = x[j];
    const Eigen::VectorXd *v2 = new Eigen::VectorXd(*v_rp);
    sampleset->erase_num(min_index);    // 删除第i个
    int n = sampleset->size();
    std::cout << "n = " << n << std::endl;
    Eigen::VectorXd k(n);
    for (int j = 0; j < n; ++j){
      k(j) = cf->get(sampleset->x(j), *v2);
    }
    double kappa = cf->get(*v2, *v2);
    if (cf->loghyper_changed){
      sampleset->insert_num(min_index, x, y);
      compute();
    } else{
      if(min_index > 0){
        L_copy.block(min_index, 0, 1, min_index) = k.head(min_index).transpose();
      }
      if (min_index < n){
        L_copy.block(min_index + 1, min_index, n - min_index, 1) = k.tail(n - min_index);
      }
      L_copy(min_index, min_index) = kappa;
      L.topLeftCorner(n+1, n+1) = L_copy.topLeftCorner(n+1, n+1).selfadjointView<Eigen::Lower>().llt().matrixL();
      sampleset->insert_num(min_index, x, y);
    }
    alpha_needs_update = true;
    delete v_rp;
    delete v2;

    int total_size = L_copy_lst.size();   // 共300个点
    // std::cout << "total_size = " << total_size << std::endl;
    Eigen::VectorXd k_hist_cir(total_size - 2);    // 除去index_hist点与被替换的点， 共298个点
    int num_cal  = 0;
    for(int index_hist = 0; index_hist < total_size; ++index_hist){ 
      if (index_hist == min_index){
        num_cal = 1;
        continue;
      }
      k_hist_cir.head(index_hist - num_cal) = k.head(index_hist - num_cal);     // k为299个点与新替换的点的核函数，k_hist_cir为k除去index_hist点，298个点
      k_hist_cir.tail(total_size - (index_hist - num_cal) - 2) = k.tail(total_size - (index_hist - num_cal) - 2);
      if (index_hist < min_index){
        if(min_index > 0){
          L_copy_lst[index_hist].block(min_index - 1, 0, 1, min_index - 1) = k_hist_cir.head(min_index - 1).transpose();
        }
        if (min_index < n){
          L_copy_lst[index_hist].block(min_index, min_index - 1, n - min_index, 1) = k_hist_cir.tail(n - min_index);
        }
        L_copy_lst[index_hist](min_index - 1, min_index - 1) = kappa;
        star_lst[index_hist](min_index - 1) = cf->get(sampleset->x(index_hist),sampleset->x(min_index));    // star中为index_hist与每一个其他的核函数；
      }else{
        if(min_index > 0){
          L_copy_lst[index_hist].block(min_index, 0, 1, min_index) = k_hist_cir.head(min_index).transpose();
        }
        if (min_index < n){
          L_copy_lst[index_hist].block(min_index + 1, min_index, n - min_index - 1, 1) = k_hist_cir.tail(n - min_index - 1);
        }
        L_copy_lst[index_hist](min_index, min_index) = kappa;
        star_lst[index_hist](min_index) = cf->get(sampleset->x(index_hist),sampleset->x(min_index));    // star中为index_hist与每一个其他的核函数；
      }
      L_lst[index_hist].topLeftCorner(n, n) = L_copy_lst[index_hist].topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
      star_hist = star_lst[index_hist];
      Eigen::VectorXd gamma_gp = L_lst[index_hist].topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(star_hist);
      sum_gamma -= gamma_lst[index_hist];
      // std::cout << "t_now - t_lst[index_hist] = " << t_now - t_lst[index_hist] << " exp = " << exp(-(t_now - t_lst[index_hist]) *(t_now - t_lst[index_hist]) / (2.0 * 250000.0)) << std::endl;
      // std::cout << "gamma_before = " << gamma_lst[index_hist] << std::endl;
      gamma_lst[index_hist] = (cf->get(sampleset->x(index_hist), sampleset->x(index_hist)) - gamma_gp.dot(gamma_gp)) * exp(-(t_now - t_lst[index_hist]) *(t_now - t_lst[index_hist]) / (2.0 *  2500000.0));
      // std::cout << "index_hist = " << index_hist << std::endl;
      // std::cout << "gamma_1 = " << cf->get(sampleset->x(index_hist), sampleset->x(index_hist)) << " gamma_2 = " << gamma_gp.dot(gamma_gp) << std::endl;
      // std::cout << "gamma_after = " << gamma_lst[index_hist] << std::endl;
       if (gamma_lst[index_hist] < 0.0){
        int flag = getchar();
       }
      sum_gamma += gamma_lst[index_hist];
    }
    t_lst[min_index] = t_now;
    gamma_lst[min_index] = gamma_new;
    sum_gamma += gamma_lst[min_index];
    average_gamma = sum_gamma / gamma_lst.size();
    std::cout << "average_gamma = " << average_gamma << std::endl;
    clock_t end_replace = clock();
    double duration_replace = (double)(end_replace - start_replace) / CLOCKS_PER_SEC;
    std::cout << "duration_replace = " << duration_replace << std::endl;
  }

  void GaussianProcess::replace_pattern_2(const int i, const double x[], double y){
    clock_t start = clock(); 
    Eigen::MatrixXd L_Last = L;
    Eigen::VectorXd *v_rp = new Eigen::VectorXd(input_dim);
    for (size_t j=0; j<input_dim; ++j) (*v_rp)(j) = x[j];
    const Eigen::VectorXd *v2 = new Eigen::VectorXd(*v_rp);
    sampleset->erase_num(i);
    int n = sampleset->size();
    if (cf->loghyper_changed){
      sampleset->insert_num(i, x, y);
      compute();
    } else{
      Eigen::VectorXd k_front(i), k_back(std::max(0, n - i));
      for (int j = 0; j < i; ++j){
        k_front(j) = cf->get(sampleset->x(j), *v2);
      }
      if (i < n){
        for (int j = 0; j < n - i; ++j){
          k_back(j) = cf->get(sampleset->x(i + j), *v2);
        }
      }
      double kappa = cf->get(*v2, *v2);
      L.topLeftCorner(i, i).triangularView<Eigen::Lower>().solveInPlace(k_front);
      L.block(i,0,1,i) = k_front.transpose();
      L(i,i) = sqrt(kappa - k_front.dot(k_front));
      sampleset->insert_num(i, x, y);
      if(i >= n){
        alpha_needs_update = true;
        return;
      }
      for (int j = i + 1; j < n + 1; ++j){
          for (int k = i; k <= j; ++k){
            if(k == i){
              L(j, k) = k_back(j - 1 - i);
              for (int index_num = 0; index_num < k; ++index_num){
                L(j, k) = L(j, k) - L(j, index_num) * L(k, index_num);
              }
              L(j, k) = L(j, k) / L(k , k);
            }
            else if(k != j) {
              L(j, k) = L_Last(j, k) * L_Last(k, k);
              for (int index_num = 0; index_num < k; ++index_num){
                L(j, k) = L(j, k) + L_Last(j, index_num) * L_Last(k, index_num) - L(j, index_num) * L(k, index_num);
              }
              L(j, k) = L(j, k) / L(k, k);
            }else{
              L(j, k) = L_Last(k, k) * L_Last(k, k);
              for (int index_num = 0; index_num < k; ++index_num){
                L(j, k) = L(j, k) + L_Last(k, index_num) * L_Last(k, index_num) - L(k, index_num) * L(k, index_num);
              }
              if (L(j, k) < 0){
                return;
              }
              L(j, k) = sqrt(L(j, k)); 
            }
          }
      }
    }
    alpha_needs_update = true;
    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "duration_replace_3 = " << duration << std::endl;
  }

  bool GaussianProcess::set_y(size_t i, double y) 
  {
    if(sampleset->set_y(i,y)) {
      alpha_needs_update = true;
      return 1;
    }
    return false;
  }

  size_t GaussianProcess::get_sampleset_size()
  {
    return sampleset->size();
  }
  
  void GaussianProcess::clear_sampleset()
  {
    sampleset->clear();
  }
  
  void GaussianProcess::write(const char * filename)
  {
    // output
    std::ofstream outfile;
    outfile.open(filename);
    time_t curtime = time(0);
    tm now=*localtime(&curtime);
    char dest[BUFSIZ]= {0};
    strftime(dest, sizeof(dest)-1, "%c", &now);
    outfile << "# " << dest << std::endl << std::endl
    << "# input dimensionality" << std::endl << input_dim << std::endl 
    << std::endl << "# covariance function" << std::endl 
    << cf->to_string() << std::endl << std::endl
    << "# log-hyperparameter" << std::endl;
    Eigen::VectorXd param = cf->get_loghyper();
    for (size_t i = 0; i< cf->get_param_dim(); i++) {
      outfile << std::setprecision(10) << param(i) << " ";
    }
    outfile << std::endl << std::endl 
    << "# data (target value in first column)" << std::endl;
    for (size_t i=0; i<sampleset->size(); ++i) {
      outfile << std::setprecision(10) << sampleset->y(i) << " ";
      for(size_t j = 0; j < input_dim; ++j) {
        outfile << std::setprecision(10) << sampleset->x(i)(j) << " ";
      }
      outfile << std::endl;
    }
    outfile.close();
  }
  
  CovarianceFunction & GaussianProcess::covf()
  {
    return *cf;
  }
  
  size_t GaussianProcess::get_input_dim()
  {
    return input_dim;
  }

  double GaussianProcess::log_likelihood()
  {
    compute();
    update_alpha();
    int n = sampleset->size();
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    double det = 2 * L.diagonal().head(n).array().log().sum();
    return -0.5*y.dot(alpha) - 0.5*det - 0.5*n*log2pi;
  }

  Eigen::VectorXd GaussianProcess::log_likelihood_gradient() 
  {
    compute();
    update_alpha();
    size_t n = sampleset->size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

    // compute kernel matrix inverse
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(W);

    W = alpha * alpha.transpose() - W;

    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j <= i; ++j) {
        cf->grad(sampleset->x(i), sampleset->x(j), g);
        if (i==j) grad += W(i,j) * g * 0.5;
        else      grad += W(i,j) * g;
      }
    }

    return grad;
  }

  double GaussianProcess::cal_gamma(Eigen::MatrixXd L_now, const Eigen::VectorXd &x_input, int index){
    gamma_star.conservativeResize(L_now.rows());
    if (L_now.rows() == 1){
      gamma_star(0) = cf->get(x_input, sampleset->x(index));
      std::cout << gamma_star(0) << std::endl;
    }
    else{
      for(int num = 0; num < L_now.rows(); ++num) {
        gamma_star(num) = cf->get(x_input, sampleset->x(num));
      }
    }
    Eigen::VectorXd gamma_gp = L_now.topLeftCorner(L_now.rows(), L_now.rows()).triangularView<Eigen::Lower>().solve(gamma_star);
    return cf->get(x_input, x_input) - gamma_gp.dot(gamma_gp);
  }

  double GaussianProcess::cal_gamma_replace(Eigen::MatrixXd L_now, const Eigen::VectorXd &x_input, int min_i){
    gamma_star.conservativeResize(L_now.rows());
    for(int num = 0; num < L_now.rows() + 1; ++num) {
      if(num == min_i){
        continue;
      }else if(num < min_i){
        gamma_star(num) = cf->get(x_input, sampleset->x(num));
      }else{
        gamma_star(num - 1) = cf->get(x_input, sampleset->x(num));
      }
    }
    Eigen::VectorXd gamma_gp = L_now.topLeftCorner(L_now.rows(), L_now.rows()).triangularView<Eigen::Lower>().solve(gamma_star);
    return cf->get(x_input, x_input) - gamma_gp.dot(gamma_gp);
  }
}
