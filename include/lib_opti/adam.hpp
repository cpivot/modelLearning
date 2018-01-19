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

  arma::vec updateVector;

public:

  Adam();
  void define(int,double,double,double,double);

  arma::vec getUpdateVector(arma::vec, double,double,arma::vec);

};

#endif
