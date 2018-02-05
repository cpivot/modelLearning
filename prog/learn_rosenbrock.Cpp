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

  model<FunctionTrain<polyElement>,Adam> model(file);

  boost::progress_display show_progress(Ntest);

  for (int ii=0;ii<Ntest;ii++)
  {
    arma::vec input=(arma::randu(DIM)-0.5)*2.;

    //double evalReal=sphere(input);
    double evalReal=rosenbrock(input);
//    std::cout << evalReal << std::endl;

    double evalModel=model(input);

    model.update(input,evalReal,(ii+1)*dt);

    sauvegarde(ii,0)=ii*dt;
    sauvegarde(ii,1)=std::abs((evalReal-evalModel)/evalReal);
   //cout << evalReal << "\t" << evalModel << endl;
    ++show_progress;
  }

  /* Statistic stuff */

  vec a = logspace<vec>(-4, 3, 50);
  uvec h1 = hist(sauvegarde(arma::span(std::floor(Ntest*2/3),Ntest-1),1), a);
  arma::mat totalHist=arma::zeros(50,2);
  totalHist.col(0)=arma::conv_to<colvec>::from(a);
  totalHist.col(1)=arma::conv_to<colvec>::from(h1);

  /* Gnuplot
  set logscale x
  set format x "%g"
  set boxwidth 0.95 relative
  set style fill transparent solid 0.5 noborder
  plot "histo.dat" u 1:2 with boxes
  */

  sauvegarde.save("save.dat",arma::raw_ascii);
  totalHist.save("histo.dat",arma::raw_ascii);

  return 0;
}
