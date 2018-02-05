#include <iostream>
#define DONT_USE_WRAPPER
#include <armadillo>
#include "model.hpp"
#include <boost/progress.hpp>


using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  string dataFile=argv[1];
  string parametersFile=argv[2];
  int colToPredict=atoi(argv[3]);

  arma::mat data;
  data.load(dataFile.c_str());
  int Nbofdata;
  if (argc==5)
    Nbofdata=atoi(argv[4]);
  else
    Nbofdata=data.n_rows;

  model<FunctionTrain<polyElement>,Adadelta> modelLearning(parametersFile);

  int ninput=modelLearning.returnnNinput();
  arma::mat sauvegarde=arma::zeros(Nbofdata,2);

  boost::progress_display show_progress(Nbofdata);

  for (int ii=0;ii<Nbofdata-1;ii++)
  {
    double time=data(ii,0);
    arma::vec input=arma::trans(data(ii,arma::span(1,ninput)));
    double real=data(ii+1,colToPredict);

    double error=modelLearning.update(input,real,time);

    sauvegarde(ii,0)=time;
    sauvegarde(ii,1)=std::abs(error/real);
    ++show_progress;
  }

  /* Statistic stuff */

  vec a = logspace<vec>(-4, 3,30);
  uvec h1 = hist(sauvegarde(arma::span(std::floor(Nbofdata*2/3),Nbofdata-1),1), a);
  arma::mat totalHist=arma::zeros(30,2);
  totalHist.col(0)=arma::conv_to<colvec>::from(a);
  totalHist.col(1)=arma::conv_to<colvec>::from(h1);


  sauvegarde.save("save.dat",arma::raw_ascii);
  totalHist.save("histo.dat",arma::raw_ascii);

  return 0;
}
