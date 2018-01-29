#include <iostream>
#include <armadillo>
#include <math.h>

#include "lib_opti.hpp"
#include "lib_base.hpp"
#include "lib_repres.hpp"

#ifndef ____MODEL__
#define ____MODEL__

template <typename representation, typename optimizer>
class model
{
private:
  representation repres;
  optimizer opti;
  double time; // Global value for time

  int numberOfParameters;

  arma::mat Bound;

public:
  model();
  model(arma::vec,int,arma::mat);
  void define(representation,optimizer,arma::mat);
  void randomizeParameters();

  double eval(arma::vec);
  double operator()(arma::vec);

  arma::vec jacobian(arma::vec);
  void update(arma::vec, double,double t=1);
};




#endif
