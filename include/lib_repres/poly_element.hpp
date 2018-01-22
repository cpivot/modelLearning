#include <iostream>
#include <armadillo>
#include <math.h>
#include <vector>
#include "lib_base.hpp"

#ifndef ____polyElement__
#define ____polyElement__

class polyElement
{
private:

  int order;
  int numberOfParameters;

  std::vector<legendre> Poly;
  std::vector<polynom> dPoly;

public:

  polyElement();
  polyElement(int);

  int returnNumberOfParameters();

  double evalElement(arma::vec,double);
  double evaldElement(arma::vec,double);
  void getParametersForGrad(arma::vec &,int,int,int,double);
  double evalGradwrtParamElement(arma::vec,double);

};

#endif
