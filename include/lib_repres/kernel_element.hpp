#include <iostream>
#include <armadillo>
#include <math.h>
#include <vector>
#include "lib_base.hpp"

#ifndef ____kernelElement__
#define ____kernelElement__

class kernelElement
{
private:

  int numberOfParameters;

public:

  kernelElement();
  void define(int,bool);

  void initialize(arma::vec &, double,int, arma::vec);

  int returnNumberOfParameters();

  double evalElement(arma::vec,double);
  double evaldElement(arma::vec,double);
  void getParametersForGrad(arma::vec &,int,int,int,double);
  double evalGradwrtParamElement(arma::vec,double);

};

#endif
