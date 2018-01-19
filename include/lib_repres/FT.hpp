#include <iostream>
#include <armadillo>
#include <math.h>
#include <list>
#include "lib_base.hpp"
#include "lib_opti.hpp"


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
  std::list<polynom> dPolyLegendre;
  std::list<arma::mat> interneMatrix;

  int whatWeEval; //1 : value ; 2 : jacobian ; 3 : gradwrtParam

  arma::vec jac;
  int currentDimJacobian;

  arma::vec gradwrtParam;
  arma::vec gradParameters();
  arma::vec parametersForGrad;


  // optimizers
  Adam Opti;

public:
  FunctionTrain();
  FunctionTrain(arma::vec, int,int,double);     //ranks,input,order
  void initialize(double);

  double eval(arma::vec);
  double operator()(arma::vec);

  arma::vec jacobian(arma::vec);
  double evalElementJacobian(arma::vec);

  arma::vec returnGradwrtParameters(arma::vec);
  void gradwrtParameters(arma::vec);
  void updateParametersForGrad(int);


  double internEval(arma::vec);
  arma::mat returnInterneMatrix(int,double); //input numer, double value
  int evalBaseIndex(int,int,int);
  double returnInterneElement(int,double);

  void update(arma::vec,double,double);

};

#endif
