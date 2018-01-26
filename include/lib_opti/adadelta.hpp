#include <iostream>
#include <armadillo>
#include <math.h>


#ifndef ____ADADELTA__
#define ____ADADELTA__

class Adadelta
{
private:
  int nparam;
  double rho;
  double signe;

  arma::vec Eg;
  arma::vec Edx;
  arma::vec dx;

public:
  Adadelta();
  Adadelta(int,double rho=0.95,double signe =1);
  arma::vec RMS(arma::vec);
  arma::vec getUpdateVector(arma::vec, double,arma::vec,double t=0);
};

#endif
