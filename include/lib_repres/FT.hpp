#include <iostream>
#include <armadillo>
#include <math.h>
#include <list>
#include "poly_element.hpp"
#include "kernel_element.hpp"
#include "lib_base.hpp"
#include "lib_opti.hpp"


#ifndef ____FT__
#define ____FT__

class FunctionTrain
{
private:
  arma::vec ranks;
  int ninput;

  int numberOfParameters;
  int numberParamElement;
  arma::vec parameters;

  std::list<arma::mat> interneMatrix;

  polyElement poly;

  int whatWeEval; //1 : value ; 2 : jacobian ; 3 : gradwrtParam

  arma::vec jac;
  int currentDimJacobian;
  int currentDeriveJacobian;


  arma::vec gradwrtParam;
  arma::vec gradParameters;
  arma::vec parametersForGrad;


  // optimizers
  Adam Opti;

public:
  FunctionTrain();
  FunctionTrain(arma::vec, int,polyElement);     //ranks,input
  void initialize(double);

  double eval(arma::vec);
  double operator()(arma::vec);

  arma::vec jacobian(arma::vec);
  double evalElementJacobian(arma::vec);

  arma::vec returnGradwrtParameters(arma::vec);
  void gradwrtParameters(arma::vec);
  void updateParametersForGrad(int,double);


  double internEval(arma::vec);
  arma::mat returnInterneMatrix(int,double); //input numer, double value
  int evalBaseIndex(int,int,int);
  double returnInterneElement(int,double);

  void update(arma::vec,double,double);

};

#endif
