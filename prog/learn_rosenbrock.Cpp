#include <iostream>
#include <math.h>
#define DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#include <armadillo>
#include "model.hpp"

#include <boost/progress.hpp>


#define DIM 5

using namespace std;
using namespace arma;

double rosenbrock(arma::vec x)
{
  double value=0;
  for (int ii=0;ii<DIM-1;ii++)
  {
    value+=100*std::pow(x(ii+1)-std::pow(x(ii),2),2);
    value+=std::pow(1-x(ii),2);
  }

  return value;
}



int main()
{
  arma_rng::set_seed_random();
  int ninput=DIM;

  int Ntest=1e5;
  double dt=1e-1;
  arma::mat sauvegarde=arma::zeros(Ntest,2);

  boost::progress_display show_progress(Ntest);


  arma::vec ranks=arma::ones(DIM-1);
  ranks << 2 << 4 << 4 << 2;

  int order=3;
  polyElement legen(order);
  FunctionTrain<polyElement> FTpoly(ranks,ninput,legen);
  Adam optiAdamPoly(FTpoly.returnNumberOfParameters(),0.01);
  Adadelta optiAdadPoly(FTpoly.returnNumberOfParameters());

  model<FunctionTrain<polyElement>,Adam> modelPolyAdam(FTpoly,optiAdamPoly);
  model<FunctionTrain<polyElement>,Adadelta> modelPolyAdadelta(FTpoly,optiAdadPoly);


  kernelElement kern;
  FunctionTrain<kernelElement> FTkernel(ranks,ninput,kern);
  Adam optiKernel(FTkernel.returnNumberOfParameters(),0.1);
  model<FunctionTrain<kernelElement>,Adam> modelKernel(FTkernel,optiKernel);


  for (int ii=0;ii<Ntest;ii++)
  {
    arma::vec input=arma::randu(DIM);
    double evalReal=rosenbrock(input);
    double evalModel=modelPolyAdadelta(input);

    modelPolyAdadelta.update(input,evalReal-evalModel,(ii+1)*dt);

    sauvegarde(ii,0)=ii*dt;
    sauvegarde(ii,1)=std::abs((evalReal-evalModel)/evalReal);
    ++show_progress;
  }

  sauvegarde.save("save.dat",arma::raw_ascii);

  return 0;
}
