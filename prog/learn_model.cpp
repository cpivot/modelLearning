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

  model<FunctionTrain<kernelElement>,Adadelta> modelLearning(parametersFile);

  int ninput=modelLearning.returnnNinput();

  boost::progress_display show_progress(Nbofdata);

  for (int ii=0;ii<Nbofdata-1;ii++)
  {
    double time=data(ii,0);
    arma::vec input=arma::trans(data(ii,arma::span(1,ninput)));
    double real=data(ii+1,colToPredict);
    double error=modelLearning.update(input,real,time);
    ++show_progress;
  }

  return 0;
}
