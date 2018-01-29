#include <iostream>
#define DONT_USE_WRAPPER
#include <armadillo>
#include "model.hpp"

using namespace std;
using namespace arma;

int main()
{
  arma_rng::set_seed_random();
  int ninput=3;

  string file="default.json";

  model<FunctionTrain<kernelElement>,Adam> modellearning(file);

  arma::vec input=arma::randu<arma::vec>(ninput);
  cout << modellearning(input) << endl;

  arma::vec testJac=modellearning.jacobian(input);
  testJac.print("Jacobian");

  double err=0.1;
  modellearning.update(input,err,0.5);

  return 0;
}
