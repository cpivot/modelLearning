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
  void define(int,double signe =1,double rho=0.95,double nonuseA=0,double nouseB=0);
  arma::vec RMS(arma::vec);
  arma::vec getUpdateVector(arma::vec, double,arma::vec,double t=0);
};

#endif
