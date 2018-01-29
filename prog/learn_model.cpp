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

  arma::mat bounds=arma::mat(ninput,2);
  bounds.col(0).fill(-1);
  bounds.col(0).fill(1);

  arma::vec ranks;
  //ranks << 2 << 4 << 3 << 2;
  ranks << 3 << 3;

  model<FunctionTrain<kernelElement>,Adam> modellearning(ranks,ninput,bounds);

  arma::vec input=arma::randu<arma::vec>(ninput);
  cout << modellearning(input) << endl;

  arma::vec testJac=modellearning.jacobian(input);
  testJac.print("Jacobian");

  double err=0.1;
  modellearning.update(input,err,0.5);

  return 0;
}
