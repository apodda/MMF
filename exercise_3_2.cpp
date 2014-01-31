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
  unsigned long int Q = 100000;

  // Exercise 3.2
  // By using the payoff and the representation in terms of an ex-
  // pectation for the option price, compute numerically with Monte Carlo meth-
  // ods the price at 0 of an Asian call option, with a 95% CI.
  // As an example, consider the same parameters as in Exercise 3.1.

//  std::function<colvec::fixed<1>() > discounted_asian_call = [S_0, T, r, sigma, K, mat_one]() -> colvec::fixed<1>
//  {
//    colvec::fixed<1> lambda = "1";
//    
//    // FIXME: Maybe with 1000 points per path it will work better
//    return options::asian_call(simulate_path<1>(S_0, T, r, sigma, 1000), lambda, K, T) * exp(-r * T) * mat_one;
//  };

//  std::function<colvec::fixed<1>() > discounted_asian_put = [S_0, T, r, sigma, K, mat_one]() -> colvec::fixed<1>
//  {
//    colvec::fixed<1> lambda = "1";
//    
//    // FIXME: Maybe with 1000 points per path it will work better
//    return options::asian_call(simulate_path<1>(S_0, T, r, sigma, 1000), lambda, K, T) * exp(-r * T) * mat_one;
//  };

  std::pair<double, double> asian_call, asian_put;

  std::cout << std::endl << std::endl;
  std::cout << "Exercise 3.2 (price of an asian call and put option):" << std::endl << std::endl;
  
  asian_call = options::asian_call_montecarlo(Q, as_scalar(S_0), T, r, as_scalar(sigma), 1000, K);
  std::cout << "Asian Call price (Montecarlo simulation): " << std::endl;
  print_pair("  Montecarlo simulation: ", asian_call);
  
  asian_put = options::asian_put_montecarlo(Q, as_scalar(S_0), T, r, as_scalar(sigma), 1000, K);
  std::cout << "Asian Put price (Montecarlo simulation): " << std::endl;
  print_pair("  Montecarlo simulation: ", asian_put);
  
  std::cout << "Difference between the price of an asian call and put is " << asian_call.first - asian_put.first;
  std::cout << ", while it should be: " << options::asian_put_call_parity(as_scalar(S_0), K, r, T) << std::endl;

  std::system("pause");
  
  return 0;
}
