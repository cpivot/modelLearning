#include <iostream>
#include <math.h>
#define DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#include <armadillo>
#include "model.hpp"
#include <boost/progress.hpp>

#define DIM 4

using namespace std;
using namespace arma;

double rosenbrock(arma::vec x)
{
  double value=0;
  for (int ii=0;ii<DIM-1;ii++)
    value+=100*std::pow(x(ii+1)-std::pow(x(ii),2),2)+std::pow(1-x(ii),2);

  return value;
}



int main(int argc, char** argv)
{
  arma_rng::set_seed_random();
  int ninput=DIM;

  int Ntest=atoi(argv[1]);
  double dt=1e-1;
  arma::mat sauvegarde=arma::zeros(Ntest,2);

  arma::mat bounds=arma::mat(DIM,2);
  bounds.col(0).fill(4); // Upper bound
  bounds.col(1).fill(-4); // Lower Bound

  boost::progress_display show_progress(Ntest);

  arma::vec ranks=arma::ones(DIM-1);
  ranks << 3 << 5 << 3;

  int order=3;
  polyElement legen(order,false);
  FunctionTrain<polyElement> FTpoly(ranks,ninput,legen);
  Adam optiAdamPoly(FTpoly.returnNumberOfParameters(),0.01);
  Adadelta optiAdadPoly(FTpoly.returnNumberOfParameters(),0.99,1);

  model<FunctionTrain<polyElement>,Adam> modelPolyAdam(FTpoly,optiAdamPoly,bounds);
  model<FunctionTrain<polyElement>,Adadelta> modelPolyAdadelta(FTpoly,optiAdadPoly,bounds);

  modelPolyAdadelta.randomizeParameters();


  kernelElement kern;
  FunctionTrain<kernelElement> FTkernel(ranks,ninput,kern);
  Adam optiAdamKernel(FTkernel.returnNumberOfParameters(),0.1);
  Adadelta optiAdadKernel(FTkernel.returnNumberOfParameters());
  model<FunctionTrain<kernelElement>,Adam> modelKernelAdam(FTkernel,optiAdamKernel,bounds);
  model<FunctionTrain<kernelElement>,Adadelta> modelKernelAdadelta(FTkernel,optiAdadKernel,bounds);


  for (int ii=0;ii<Ntest;ii++)
  {
    arma::vec input=(arma::randu(DIM)-0.5)*6.;

    double evalReal=rosenbrock(input);
    double evalModel=modelPolyAdadelta(input);

    modelPolyAdadelta.update(input,2*(evalModel-evalReal),(ii+1)*dt);

    sauvegarde(ii,0)=ii*dt;
    sauvegarde(ii,1)=std::abs((evalReal-evalModel)/evalReal);
//    cout << evalReal << "\t" << evalModel << endl;
    ++show_progress;
  }

  sauvegarde.save("save.dat",arma::raw_ascii);

  return 0;
}
