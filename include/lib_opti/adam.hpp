#include <iostream>
#include <armadillo>
#include <math.h>


#ifndef ____ADAM__
#define ____ADAM__

class Adam
{
private:

  int nparam;

  double alpha;
  double betaun;
  double betadeux;

  double signe;

  arma::vec m;
  arma::vec v;
  arma::vec mBias;
  arma::vec vBias;

public:
  Adam();
  void define(int,double,double betaun=0.9,double betadeux=0.999,double signe =1);
  arma::vec getUpdateVector(arma::vec, double,arma::vec,double t=1);
};

#endif
