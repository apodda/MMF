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
  
  //Exercise 3.6. In the case of a call option, implement the dynamic hedging
  //described above.
  //Consider the parameters as in Exercise 3.1 and the formulas for the price
  //and the delta as in (5) and (6) respectively.
  
  std::cout << std::endl << std::endl;
  std::cout << "Exercise 3.6 (implementing a monthly hedging strategy, using exact formulas for the price and the delta):" << std::endl << std::endl;
  
  // Choose a drift different from r
  double drift = 0.5;
  std::vector<colvec::fixed<1> > path = simulate_path(S_0, T, drift, sigma, 12); // Monthly monitoring
  double Z = fmax(as_scalar(path.back()) - K, 0); // Payoff
  unsigned int width = 8; // Formatting
  
  std::function<double(colvec::fixed<1>, double) > bound_explicit_call_prize = [T, r, sigma, K](colvec::fixed<1> x, double t) -> double
  {
    return explicit_call_price(as_scalar(x), T - t, K, r, as_scalar(sigma));
  };
  
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > bound_explicit_call_delta = [T, r, sigma, K, mat_one](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    return explicit_call_delta(as_scalar(x), T - t, K, r, as_scalar(sigma)) * mat_one;
  };
  
  std::vector<colvec::fixed<3> > hedging_strategy = dynamic_hedging<1>(path, bound_explicit_call_prize, bound_explicit_call_delta, T, K, r, sigma);
  std::cout << "Monthly Hedging Strategy (exact formulas):" << std::endl;
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
