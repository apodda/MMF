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

  // Exercise 3.1
  // Compute numerically, by means of the Monte Carlo method,
  // the price of a call and of a put option at time 0, with a 95% CI. Moreover,
  // study (empirically) the convergence to the true prices when the simulation
  // number Q tends to infinity.
  // As an example, consider the following parameters: S 0 = K = 100, T = 1
  // year, r = 0.05, σ = 0.2.
  
  double exact_call, exact_put;
  
  std::cout << "Exercise 3.1 (price of an european call and put option):" << std::endl << std::endl;
  
  std::function<colvec::fixed<1>() > discounted_european_call = [S_0, T, r, sigma, K, mat_one]() -> colvec::fixed<1>
  {
    colvec::fixed<1> gaussian;
    colvec::fixed<1> lambda = "1";
    
    gaussian.randn();
    return options::european_call(step<1>(gaussian, S_0, T, r, sigma), lambda, K) * exp(-r * T) * mat_one; 
  };
  
  std::function<colvec::fixed<1>() > discounted_european_put = [S_0, T, r, sigma, K, mat_one]() -> colvec::fixed<1>
  {
    colvec::fixed<1> gaussian;
    colvec::fixed<1> lambda = "1";
    
    gaussian.randn();
    return options::european_put(step<1>(gaussian, S_0, T, r, sigma), lambda, K) * exp(-r * T) * mat_one; 
  };
  
  std::cout << "European Call price:" << std::endl;
  print_pair("  Montecarlo simulation: ", montecarlo<1>(Q, discounted_european_call));

  // t=0 => time_step = T - t = T
  exact_call = explicit_call_price(as_scalar(S_0), T, K, r, as_scalar(sigma));
  std::cout << "  Exact formula: " << exact_call << std::endl;
  
  std::cout << "European Put price:" << std::endl;
  print_pair("  Montecarlo simulation: ", montecarlo<1>(Q, discounted_european_put));
     
  // t=0 => time_step = T - t = T
  exact_put = explicit_put_price(as_scalar(S_0), T, K, r, as_scalar(sigma));
  std::cout << "  Exact formula: " << exact_put << std::endl;

  convergence_study(discounted_european_call, "European_Call", exact_call);
  std::cout << "Simulations/error graphic of the european call written as European_Call.png" << std::endl;
  convergence_study(discounted_european_put, "European_Put", exact_put);
  std::cout << "Simulations/error graphic of the european put written as European_Put.png" << std::endl;

  return 0;
}