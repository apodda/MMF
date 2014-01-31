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
  
  // Choose a drift different from r
  double drift = 1;
  std::vector<colvec::fixed<1> > path = simulate_path(S_0, T, drift, sigma, 12); // Monthly monitoring
  double Z = fmax(as_scalar(path.back()) - K, 0); // Payoff
  unsigned int width = 8; // Formatting

  std::vector<colvec::fixed<3> > hedging_strategy;
  
  //Exercise 3.7. Repeat Exercise 3.6 by using a Monte Carlo method to com-
  //pute the initial price V_0 and the delta (that is, not using the exact formulas).
  //As for the delta, split in two codes: one making use of the technique in Section
  //3.2.1 and the other with the method in Section 3.2.2

  std::cout << std::endl << std::endl;
  std::cout << "Exercise 3.7 (implementing a monthly hedging strategy, using Montecarlo simulation for the price and the delta):" << std::endl << std::endl;

  // Function to calculate the price via montecarlo simulation
  auto bound_ec = std::bind(options::european_call<1>, _1, mat_one, _2);
  
  // Function to calculate the delta via finite differences
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > call_fd_delta = [T, r, sigma, K, delta, bound_ec](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    return finite_difference_delta<1>(x, T - t, r, sigma, bound_ec, K, delta); 
  };
  
  // Function to calculate the delta with the correction terms obtained with the representation formula
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > call_m_delta = [T, r, sigma, K, delta, bound_ec](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    return malliavin_delta<1>(x, T - t, r, sigma, bound_ec, K); 
  };
  
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > montecarlo_call_fd_delta = [&call_fd_delta, Q](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    std::function<colvec::fixed<1>() > bound_call_fd_delta = std::bind(call_fd_delta, x, t);
    
    return montecarlo<1>(Q, bound_call_fd_delta).first;
  };
  
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > montecarlo_call_m_delta = [&call_m_delta, Q](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    std::function<colvec::fixed<1>() > bound_call_m_delta = std::bind(call_m_delta, x, t);
    
    return montecarlo<1>(Q, bound_call_m_delta).first;
  };
  
  std::function<double(colvec::fixed<1>, double) > montecarlo_call_price = [&](colvec::fixed<1> x, double t) -> double
  {
    std::function<colvec::fixed<1>() > european_call = [&]() -> colvec::fixed<1>
    {
      colvec::fixed<1> gaussian;
      colvec::fixed<1> lambda = "1";
      
      gaussian.randn();
      return options::european_call(step<1>(gaussian, x, T - t, r, sigma), lambda, K) * mat_one; 
    };
    
    return as_scalar(montecarlo<1>(Q, european_call).first);
  };
  
  
  
  std::cout << "Monthly Hedging Strategy (montecarlo, finite differences):" << std::endl;
  hedging_strategy = dynamic_hedging<1>(path, montecarlo_call_price, montecarlo_call_fd_delta, T, K, r, sigma);

  std::cout << "Delta_0  Delta_1  V_t" << std::endl;
  for(auto iter = hedging_strategy.begin(); iter != hedging_strategy.end(); ++iter)
  {
    std::cout << std::setw(width) << (*iter)(0) << " "
              << std::setw(width) << (*iter)(1) << " " 
              << std::setw(width) << (*iter)(2) << std::endl;
  }
  
  std::cout << "The value of the portfolio at time T is " << hedging_strategy.back()(2) 
            << " while the option payoff is " << Z << std::endl;
  
  std::cout << std::endl;
  std::cout << "Monthly Hedging Strategy (montecarlo, representation formula):" << std::endl;
  hedging_strategy = dynamic_hedging<1>(path, montecarlo_call_price, montecarlo_call_m_delta, T, K, r, sigma);
  
  std::cout << "Delta_0  Delta_1  V_t" << std::endl;
  for(auto iter = hedging_strategy.begin(); iter != hedging_strategy.end(); ++iter)
  {
    std::cout << std::setw(width) << (*iter)(0) << " "
              << std::setw(width) << (*iter)(1) << " " 
              << std::setw(width) << (*iter)(2) << std::endl;
  }
  
  std::cout << "The value of the portfolio at time T is " << hedging_strategy.back()(2) 
            << " while the option payoff is " << Z << std::endl;

  std::system("pause");
  return 0;
}
