#include <iostream>
#include <armadillo>
#include <math.h>

#include "poly_element.hpp"
#include "kernel_element.hpp"
#include "lib_base.hpp"


#ifndef ____FT__
#define ____FT__

template <typename element>
class FunctionTrain
{
private:
  arma::vec ranks;
  int ninput;

  int numberOfParameters;
  int numberParamElement;
  arma::vec parameters;

  element elem;

  int whatWeEval; //1 : value ; 2 : jacobian ; 3 : gradwrtParam

  arma::vec jac;
  int currentDimJacobian;
  int currentDeriveJacobian;


  arma::vec gradwrtParam;
  arma::vec gradParameters;
  arma::vec parametersForGrad;


public:
  FunctionTrain();
  FunctionTrain(arma::vec, int,element);     //ranks,input
  void initialize(double);

  int returnNumberOfParameters();

  double eval(arma::vec);
  double operator()(arma::vec);

  arma::vec jacobian(arma::vec);
  double evalElementJacobian(arma::vec);

  void randomize();

  arma::vec returnGradwrtParameters(arma::vec);
  void gradwrtParameters(arma::vec);
  void updateParametersForGrad(int,arma::vec);


  double internEval(arma::vec);
  arma::mat returnInterneMatrix(int,double,int &); //input numer, double value
  double returnInterneElement(int,double);

  void updateParameters(arma::vec);

};



#endif
