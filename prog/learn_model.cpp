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

  arma::vec ranks;
  //ranks << 2 << 4 << 3 << 2;
  ranks << 3 << 3;

  int order=2;
  polyElement legen(order);

  kernelElement kern;
  FunctionTrain<kernelElement> FTkernel(ranks,ninput,kern);
  Adam opti(kern.returnNumberOfParameters(),0.1);

  model<FunctionTrain<kernelElement>,Adam> modellearning;
  modellearning.define(FTkernel,opti);

  arma::vec input=arma::randu<arma::vec>(ninput);
  cout << modellearning(input) << endl;

  arma::vec testJac=FTkernel.jacobian(input);
  testJac.print("Jacobian");

  return 0;
}
