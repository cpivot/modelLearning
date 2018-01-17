#include <iostream>
#include <armadillo>
#include <math.h>
#include <list>
#include "polynome.hpp"


#ifndef ____FT__
#define ____FT__

class FunctionTrain
{
private:
  arma::vec ranks;
  int order;
  int ninput;

  int numberOfParameters;
  arma::vec parameters;

  std::list<legendre> PolyLegendre;
  std::list<arma::mat> interneMatrix;

public:
  FunctionTrain(arma::vec, int,int,double);     //ranks,input,order
  void initialize(double);

  double eval(arma::vec);

  arma::mat returnInterneMatrix(int,double); //input numer, double value
  int evalBaseIndex(int,int,int);
  double returnInterneElement(int,double);

};

#endif
