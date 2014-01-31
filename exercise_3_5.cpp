#include <iostream> // std::cout
#include <fstream> // std::ofstream
#include <iomanip> // std::setw [needed for output formatting]
#include <functional> // std::function
#include <string> // std::string
#include <cmath> // pow, exp

#include <armadillo> // colvec, mat

#include "common.cpp" // Lots of stuff

using namespace arma;
using namespace std::placeholders;

int main()
{
  colvec::fixed<1> S_0 = "100";
  double K = 100;
  double T = 1;
  double r = 0.05;
  mat::fixed<1, 1> sigma = "0.2";

  // Utility: 1x1 matrix
  const colvec::fixed<1> mat_one = ones(1, 1);
  
  // Number of simulations
  unsigned int Q = 10000;
  double delta = 0.001;
  
  colvec::fixed<2> S_0_pair;
  
  S_0_pair << 100 << endr << 100 << endr;
  double rho = 0.2;
  double sigma_1 = 0.2;
  double sigma_2 = 0.2;
  
  mat::fixed<2, 2> sigma_rho;
  sigma_rho << sigma_1 << 0 << endr
            << rho << sigma_2 * sqrt(1 - rho * rho) << endr;
  
  // Exercise 3.5
  // Compute numerically, with a Monte Carlo method, the delta
  // of the options studied in Exercise 3.3 (change option and digital option on
  // two assets), with a 95% CI, either with the finite difference method and with
  // the representation formula for the derivative.
  // As for the parameters, see Exercise 3.1 and for the finite differences, take for
  // example δ = S 0 10^−3 .

  std::cout << std::endl << std::endl;
  std::cout << "Exercise 3.5 (delta of an exchange and digital option):" << std::endl << std::endl;

  std::pair<colvec::fixed<2>, colvec::fixed<2> > tmp_delta;
  
  colvec::fixed<2> lambda = "1 -1";
  colvec::fixed<2> lambda_flip = "-1 1";

  auto bound_c = std::bind(options::european_call<2>, _1, lambda, _2);
  auto bound_c_flip = std::bind(options::european_call<2>, _1, lambda_flip, _2);
  auto bound_d = std::bind(options::digital<2>, _1, lambda, std::placeholders::_2);
  auto bound_d_flip = std::bind(options::digital<2>, _1, lambda_flip, _2);

  // K = 0 !
  std::function<colvec::fixed<2>(colvec::fixed<2>, double) > exchange_fd_delta = [&](colvec::fixed<2> x, double t) -> colvec::fixed<2>
  {
    return finite_difference_delta<2>(x, T - t, r, sigma_rho, bound_c, 0, delta);
  };
  
  std::function<colvec::fixed<2>(colvec::fixed<2>, double) > exchange_fd_delta_flip = [&](colvec::fixed<2> x, double t) -> colvec::fixed<2>
  {
    return finite_difference_delta<2>(x, T - t, r, sigma_rho, bound_c_flip, 0, delta);
  };

  std::function<colvec::fixed<2>(colvec::fixed<2>, double) > exchange_m_delta = [&](colvec::fixed<2> x, double t) -> colvec::fixed<2>
  {
    return malliavin_delta<2>(x, T - t, r, sigma_rho, bound_c, 0);
  };
  
  std::function<colvec::fixed<2>(colvec::fixed<2>, double) > exchange_m_delta_flip = [&](colvec::fixed<2> x, double t) -> colvec::fixed<2>
  {
    return malliavin_delta<2>(x, T - t, r, sigma_rho, bound_c_flip, 0);
  };

  std::function<colvec::fixed<2>(colvec::fixed<2>, double) > digital_fd_delta = [&](colvec::fixed<2> x, double t) -> colvec::fixed<2>
  {
    return finite_difference_delta<2>(x, T - t, r, sigma_rho, bound_d, 0, delta);
  };
  
  std::function<colvec::fixed<2>(colvec::fixed<2>, double) > digital_fd_delta_flip = [&](colvec::fixed<2> x, double t) -> colvec::fixed<2>
  {
    return finite_difference_delta<2>(x, T - t, r, sigma_rho, bound_d_flip, 0, delta);
  };

  std::function<colvec::fixed<2>(colvec::fixed<2>, double) > digital_m_delta = [&](colvec::fixed<2> x, double t) -> colvec::fixed<2>
  {
    return malliavin_delta<2>(x, T - t, r, sigma_rho, bound_d, 0);
  };  
  
  std::function<colvec::fixed<2>(colvec::fixed<2>, double) > digital_m_delta_flip = [&](colvec::fixed<2> x, double t) -> colvec::fixed<2>
  {
    return malliavin_delta<2>(x, T - t, r, sigma_rho, bound_d_flip, 0);
  };

  std::function<colvec::fixed<2>() > bound_exchange_fd_delta = std::bind(exchange_fd_delta, S_0_pair, 0);
  std::function<colvec::fixed<2>() > bound_exchange_m_delta = std::bind(exchange_m_delta, S_0_pair, 0);
  std::function<colvec::fixed<2>() > bound_exchange_fd_delta_flip = std::bind(exchange_fd_delta_flip, S_0_pair, 0);
  std::function<colvec::fixed<2>() > bound_exchange_m_delta_flip = std::bind(exchange_m_delta_flip, S_0_pair, 0);
  
  std::function<colvec::fixed<2>() > bound_digital_fd_delta = std::bind(digital_fd_delta, S_0_pair, 0);
  std::function<colvec::fixed<2>() > bound_digital_m_delta = std::bind(digital_m_delta, S_0_pair, 0);
  std::function<colvec::fixed<2>() > bound_digital_fd_delta_flip = std::bind(digital_fd_delta_flip, S_0_pair, 0);
  std::function<colvec::fixed<2>() > bound_digital_m_delta_flip = std::bind(digital_m_delta_flip, S_0_pair, 0);
  
  std::cout << "Rho = 0.2" << std::endl;
  tmp_delta = montecarlo(Q, bound_exchange_fd_delta);
  std::cout << "Exchange delta (S_1 - S_2)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_m_delta);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_fd_delta_flip);
  std::cout << "Exchange delta (S_2 - S_1)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_m_delta_flip);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;

  tmp_delta = montecarlo(Q, bound_digital_fd_delta);
  std::cout << "Digital delta (S_1 - S_2)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_m_delta);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_fd_delta_flip);
  std::cout << "Digital delta (S_2 - S_1)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_m_delta_flip);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;  
  
  
  rho = 0;
  sigma_rho << sigma_1 << 0 << endr
            << rho << sigma_2 * sqrt(1 - rho * rho) << endr;
  std::cout << std::endl << "Rho = 0" << std::endl;
  tmp_delta = montecarlo(Q, bound_exchange_fd_delta);
  std::cout << "Exchange delta (S_1 - S_2)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_m_delta);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_fd_delta_flip);
  std::cout << "Exchange delta (S_2 - S_1)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_m_delta_flip);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;

  tmp_delta = montecarlo(Q, bound_digital_fd_delta);
  std::cout << "Digital delta (S_1 - S_2)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_m_delta);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_fd_delta_flip);
  std::cout << "Digital delta (S_2 - S_1)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_m_delta_flip);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;

  rho = -0.2;
  sigma_rho << sigma_1 << 0 << endr
            << rho << sigma_2 * sqrt(1 - rho * rho) << endr;  
  std::cout << std::endl << "Rho = -0.2" << std::endl;
  tmp_delta = montecarlo(Q, bound_exchange_fd_delta);
  std::cout << "Exchange delta (S_1 - S_2)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_m_delta);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_fd_delta_flip);
  std::cout << "Exchange delta (S_2 - S_1)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_exchange_m_delta_flip);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;

  tmp_delta = montecarlo(Q, bound_digital_fd_delta);
  std::cout << "Digital delta (S_1 - S_2)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_m_delta);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_fd_delta_flip);
  std::cout << "Digital delta (S_2 - S_1)_+" << std::endl;
  std::cout << "  Finite differences:     " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  tmp_delta = montecarlo(Q, bound_digital_m_delta_flip);
  std::cout << "  Representation formula: " << tmp_delta.first(0) << " with confidence interval [" 
            << tmp_delta.first(0) - tmp_delta.second(0) << ", " 
			<< tmp_delta.first(0) + tmp_delta.second(0) << "]" << std::endl
			<< "                          " << tmp_delta.first(1) << " with confidence interval [" 
            << tmp_delta.first(1) - tmp_delta.second(1) << ", " 
			<< tmp_delta.first(1) + tmp_delta.second(1) << "]" << std::endl;
  
  std::system("pause");
  return 0;
}
