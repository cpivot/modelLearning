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
  Adam opti(FTkernel.returnNumberOfParameters(),0.1);
  model<FunctionTrain<kernelElement>,Adam> modellearning(FTkernel,opti);

  decltype(modellearning) toto(FTkernel,opti);


  arma::vec input=arma::randu<arma::vec>(ninput);
  cout << modellearning(input) << endl;

  arma::vec testJac=modellearning.jacobian(input);
  testJac.print("Jacobian");

  double err=0.1;
  modellearning.update(input,err,0.5);

  cout << toto(input) << endl;
  return 0;
}
