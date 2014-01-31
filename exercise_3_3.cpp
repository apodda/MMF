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


  // Exercise 3.3
  // By using (simply) the payoff and the representation of the
  // price in terms of an expectation, compute numerically, by a Monte Carlo
  // method, the price at 0 of the options with payoff given in (7), with a 95% CI.
  // As an example, consider parameters similar to the ones suggested in Exercise
  // 3.1 and study the case ρ = 0 (independence) and ρ = ±0.2 (positive/negative
  // dependence).
  
  std::cout << std::endl << std::endl;
  std::cout << "Exercise 3.2 (price of an exchange and digital option):" << std::endl << std::endl;

  colvec::fixed<2> S_0_pair;
  
  S_0_pair << 100 << endr << 100 << endr;
  double rho = 0.2;
  double sigma_1 = 0.2;
  double sigma_2 = 0.2;
  
  std::pair<colvec::fixed<1>, colvec::fixed<1> > digital, digital_flip, exchange, exchange_flip;

  mat::fixed<2, 2> sigma_rho;
  sigma_rho << sigma_1 << 0 << endr
            << rho << sigma_2 * sqrt(1 - rho * rho) << endr;

  std::function<colvec::fixed<1>() > discounted_exchange = [S_0_pair, T, r, K, &sigma_rho, mat_one]() -> colvec::fixed<1>
  {
    colvec::fixed<2> gaussian;
    colvec::fixed<2> lambda = "1 -1";
    
    gaussian.randn();
    // Payoff: (S^1 - S^2)_+ => K = 0
    return options::european_call(step<2>(gaussian, S_0_pair, T, r, sigma_rho), lambda, 0) * exp(-r * T) * mat_one; 
  };
  
  std::function<colvec::fixed<1>() > discounted_exchange_flip = [S_0_pair, T, r, K, &sigma_rho, mat_one]() -> colvec::fixed<1>
  {
    colvec::fixed<2> gaussian;
    colvec::fixed<2> lambda = "-1 1";
    
    gaussian.randn();
    // Payoff: (S^2 - S^1)_+ => K = 0
    return options::european_call(step<2>(gaussian, S_0_pair, T, r, sigma_rho), lambda, 0) * exp(-r * T) * mat_one; 
  };
  
  std::function<colvec::fixed<1>() > discounted_digital = [S_0_pair, T, r, K, &sigma_rho, mat_one]() -> colvec::fixed<1>
  {
    colvec::fixed<2> gaussian;
    colvec::fixed<2> lambda = "1 -1";
    
    gaussian.randn();
    // Payoff: (S^1 - S^2)_+ => K = 0
    return options::digital(step<2>(gaussian, S_0_pair, T, r, sigma_rho), lambda, 0) * exp(-r * T) * mat_one; 
  };
  
  std::function<colvec::fixed<1>() > discounted_digital_flip = [S_0_pair, T, r, K, &sigma_rho, mat_one]() -> colvec::fixed<1>
  {
    colvec::fixed<2> gaussian;
    colvec::fixed<2> lambda = "-1 1";
    
    gaussian.randn();
    // Payoff: (S^2 - S^1)_+ => K = 0
    return options::digital(step<2>(gaussian, S_0_pair, T, r, sigma_rho), lambda, 0) * exp(-r * T) * mat_one; 
  };
  
  // rho = 0.2
  std::cout << "Rho = 0.2, Montecarlo simulation" << std::endl;
  
  exchange = montecarlo(Q, discounted_exchange);
  exchange_flip = montecarlo(Q, discounted_exchange_flip);
  digital = montecarlo(Q, discounted_digital);
  digital_flip = montecarlo(Q, discounted_digital_flip);

  print_pair("  Exchange option price, (S_1 - S_2)_+: ", exchange);
  print_pair("  Exchange option price (S_2 - S_1)_+: ", exchange_flip);
  print_pair("  Digital option price (S_1 - S_2)_+: ", digital);
  print_pair("  Digital option price (S_2 - S_1)_+: ", digital_flip);
  
  // Parity formula: exchange - exchange_flip = S_0^1 - S_0^2 = 0 since S_0^1 = S_0^2
  std::cout << "The difference between the two exchange options is " 
            << as_scalar(exchange.first - exchange_flip.first)
            << ", while it should be " << 0 << std::endl;

  // Parity formula: digital + digital_flip = exp(-r * T)
  std::cout << "The difference between the two digital options is " 
            << as_scalar(digital.first + digital_flip.first)
            << ", while it should be " << exp(-r * T) << std::endl;
  
  // Rho = -0.2
  std::cout << std::endl;
  std::cout << "Rho = -0.2, Montecarlo simulation" << std::endl;
  rho = -0.2;
  sigma_rho << sigma_1 << 0 << endr
            << rho << sigma_2 * sqrt(1 - rho * rho) << endr;

  exchange = montecarlo(Q, discounted_exchange);
  exchange_flip = montecarlo(Q, discounted_exchange_flip);
  digital = montecarlo(Q, discounted_digital);
  digital_flip = montecarlo(Q, discounted_digital_flip);

  print_pair("  Exchange option price, (S_1 - S_2)_+: ", exchange);
  print_pair("  Exchange option price (S_2 - S_1)_+: ", exchange_flip);
  print_pair("  Digital option price (S_1 - S_2)_+: ", digital);
  print_pair("  Digital option price (S_2 - S_1)_+: ", digital_flip);

  // Parity formula: exchange - exchange_flip = S_0^1 - S_0^2 = 0 since S_0^1 = S_0^2
  std::cout << "The difference between the two exchange options is " 
            << as_scalar(exchange.first - exchange_flip.first)
            << ", while it should be " << 0 << std::endl;

  // Parity formula: digital + digital_flip = exp(-r * T)
  std::cout << "The difference between the two digital options is " 
            << as_scalar(digital.first + digital_flip.first)
            << ", while it should be " << exp(-r * T) << std::endl;
  
  // rho = 0; 
  std::cout << std::endl;
  std::cout << "Rho = 0, Montecarlo simulation" << std::endl;
  sigma_rho << sigma_1 << 0 << endr
            << 0 << sigma_2  << endr;

  exchange = montecarlo(Q, discounted_exchange);
  exchange_flip = montecarlo(Q, discounted_exchange_flip);
  digital = montecarlo(Q, discounted_digital);
  digital_flip = montecarlo(Q, discounted_digital_flip);

  print_pair("  Exchange option price, (S_1 - S_2)_+: ", exchange);
  print_pair("  Exchange option price (S_2 - S_1)_+: ", exchange_flip);
  print_pair("  Digital option price (S_1 - S_2)_+: ", digital);
  print_pair("  Digital option price (S_2 - S_1)_+: ", digital_flip);
  
  // Parity formula: exchange - exchange_flip = S_0^1 - S_0^2 = 0 since S_0^1 = S_0^2
  std::cout << "The difference between the two exchange options is " 
            << as_scalar(exchange.first - exchange_flip.first)
            << ", while it should be " << 0 << std::endl;

  // Parity formula: digital + digital_flip = exp(-r * T)
  std::cout << "The difference between the two digital options is " 
            << as_scalar(digital.first + digital_flip.first)
            << ", while it should be " << exp(-r * T) << std::endl;

    std::system("pause");

  return 0;
}
