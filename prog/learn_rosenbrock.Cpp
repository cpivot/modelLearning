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

double sphere(arma::vec x)
{
  double value=0;
  for (int ii=0;ii<DIM;ii++)
    value+=std::pow(x(ii),2);

  return value;
}

int main(int argc, char** argv)
{
  arma_rng::set_seed_random();
  int ninput=DIM;

  int Ntest=atoi(argv[1]);
  double dt=1e-1;

  arma::mat sauvegarde=arma::zeros(Ntest,2);

  string file="../../prog/default.json";

  model<FunctionTrain<polyElement>,Adadelta> model(file);
  model.randomizeParameters();

  boost::progress_display show_progress(Ntest);

  for (int ii=0;ii<Ntest;ii++)
  {
    arma::vec input=(arma::randu(DIM)-0.5)*6.;

    double evalReal=sphere(input);
  //  double evalReal=sphere(input);
    double evalModel=model(input);

    model.update(input,2*(evalModel-evalReal),(ii+1)*dt);

    sauvegarde(ii,0)=ii*dt;
    sauvegarde(ii,1)=std::abs((evalReal-evalModel)/evalReal);
//    cout << evalReal << "\t" << evalModel << endl;
    ++show_progress;
  }

  sauvegarde.save("save.dat",arma::raw_ascii);

  return 0;
}
