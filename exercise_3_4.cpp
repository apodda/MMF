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

  // Exercise 3.4
  // Compute numerically, by a Monte Carlo method, the delta
  // of a standard call and put option at 0, with a 95% CI, either with the fi-
  // nite difference method and with the representation formula for the derivative.
  // Moreover, study empirically the convergence to the exact value as the number
  // of simulated paths Q increases to infinity.
  // As for the parameters, see Exercise 3.1 and for the finite differences, take for
  // example δ = S_0 10^−3 .

  std::cout << std::endl << std::endl;
  std::cout << "Exercise 3.4 (delta of an european call or put option):" << std::endl << std::endl;

  double delta = as_scalar(S_0) * 0.001;
  
  double call_delta_exact, put_delta_exact;
  std::pair<colvec::fixed<1>, colvec::fixed<1> > call_fd_delta_result, put_fd_delta_result;
  std::pair<colvec::fixed<1>, colvec::fixed<1> > call_m_delta_result, put_m_delta_result;
  
  // Functions that calculate payoffs

  auto bound_ec = std::bind(options::european_call<1>, _1, mat_one, _2);
  auto bound_ep = std::bind(options::european_put<1>, _1, mat_one, _2);

  // Functions that calculate the delta via finite differences
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > call_fd_delta = [T, r, sigma, K, delta, bound_ec](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    return finite_difference_delta<1>(x, T - t, r, sigma, bound_ec, K, delta); 
  };
  
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > put_fd_delta = [T, r, sigma, K, delta, bound_ep](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    return finite_difference_delta<1>(x, T - t, r, sigma, bound_ep, K, delta); 
  };

  std::function<colvec::fixed<1>() > bound_call_fd_delta = std::bind(call_fd_delta, S_0, 0);
  std::function<colvec::fixed<1>() > bound_put_fd_delta = std::bind(put_fd_delta, S_0, 0);
  
  // Functions that calculate the delta with the correction terms obtained with the representation formula
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > call_m_delta = [T, r, sigma, K, delta, bound_ec](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    return malliavin_delta<1>(x, T - t, r, sigma, bound_ec, K); 
  };
  
  std::function<colvec::fixed<1>(colvec::fixed<1>, double) > put_m_delta = [T, r, sigma, K, delta, bound_ep](colvec::fixed<1> x, double t) -> colvec::fixed<1>
  {
    return malliavin_delta<1>(x, T - t, r, sigma, bound_ep, K); 
  };
  
  std::function<colvec::fixed<1>() > bound_call_m_delta = std::bind(call_m_delta, S_0, 0);
  std::function<colvec::fixed<1>() > bound_put_m_delta = std::bind(put_m_delta, S_0, 0);
  
  // Output
  
  call_delta_exact = explicit_call_delta(as_scalar(S_0), T, K, r, as_scalar(sigma));
  put_delta_exact = explicit_put_delta(as_scalar(S_0), T, K, r, as_scalar(sigma));
  
  call_fd_delta_result = montecarlo(Q, bound_call_fd_delta);
  put_fd_delta_result = montecarlo(Q, bound_call_fd_delta);
  
  call_m_delta_result = montecarlo(Q, bound_call_m_delta);
  put_m_delta_result = montecarlo(Q, bound_call_m_delta);
  
  std::cout << "Call delta" << std::endl;
  std::cout << "  Exact formula: " << call_delta_exact << std::endl;
  print_pair("  Finite differences: ", call_fd_delta_result);
  print_pair("  Representation formula: ", call_m_delta_result);
  
  std::cout << "Put delta" << std::endl;
  std::cout << "  Exact formula: " << put_delta_exact << std::endl;
  print_pair("  Finite differences: ", put_fd_delta_result);
  print_pair("  Representation formula: ", put_m_delta_result);
  
  convergence_study(bound_call_fd_delta, "European_Call_Delta (finite differences)", call_delta_exact);
  convergence_study(bound_put_fd_delta, "European_Put_Delta (finite differences)", put_delta_exact);
  convergence_study(bound_call_m_delta, "European_Call_Delta (representation formula)", call_delta_exact);
  convergence_study(bound_put_m_delta, "European_Put_Delta (representation formula)", put_delta_exact);

  std::system("pause");
  
  return 0;
}
