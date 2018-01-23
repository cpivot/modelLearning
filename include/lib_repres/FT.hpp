#include <iostream>
#include <armadillo>
#include <math.h>

#include "poly_element.hpp"
#include "kernel_element.hpp"
#include "lib_base.hpp"
#include "lib_opti.hpp"


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


  // optimizers
  Adam Opti;

public:
  FunctionTrain();
  FunctionTrain(arma::vec, int,element);     //ranks,input
  void initialize(double);

  double eval(arma::vec);
  double operator()(arma::vec);

  arma::vec jacobian(arma::vec);
  double evalElementJacobian(arma::vec);

  arma::vec returnGradwrtParameters(arma::vec);
  void gradwrtParameters(arma::vec);
  void updateParametersForGrad(int,arma::vec);


  double internEval(arma::vec);
  arma::mat returnInterneMatrix(int,double); //input numer, double value
  int evalBaseIndex(int,int,int);
  double returnInterneElement(int,double);

  void update(arma::vec,double,double);

};






template <typename element>
FunctionTrain<element>::FunctionTrain()
{

}

template <typename element>
FunctionTrain<element>::FunctionTrain(arma::vec ranks_m,int ninput_m, element elem_m)
{
  ninput=ninput_m;
  ranks=arma::ones(ninput+1);
  ranks.subvec(1,ninput-1)=ranks_m;

  elem=elem_m;
  numberParamElement=elem.returnNumberOfParameters();

  jac=arma::zeros(ninput);

  //Evaluate number of parameters
  numberOfParameters=0;
  for (int ii=0;ii<ninput;ii++)
    numberOfParameters+=ranks(ii)*ranks(ii+1);

  numberOfParameters*=numberParamElement;
  parameters=arma::zeros(numberOfParameters);

  parametersForGrad=arma::zeros(numberOfParameters);
  gradwrtParam=arma::zeros(numberOfParameters);

  initialize(0.1);

  Opti.define(numberOfParameters,0.1,0.99,0.999,1);
}

template <typename element>
void FunctionTrain<element>::initialize(double initialValue)
{
  elem.initialize(parameters,initialValue,ninput,ranks);
}


template <typename element>
double FunctionTrain<element>::eval(arma::vec input)
{
  whatWeEval=1;
  return internEval(input);
}

template <typename element>
double FunctionTrain<element>::operator()(arma::vec input)
{
  return this->eval(input);
}



template <typename element>
arma::vec FunctionTrain<element>::jacobian(arma::vec input)
{
  whatWeEval=2;

  for (int ii=0;ii<ninput;ii++)
  {
    currentDimJacobian=ii;
    jac(ii)=evalElementJacobian(input);
  }

  return jac;
}

template <typename element>
double FunctionTrain<element>::evalElementJacobian(arma::vec input)
{
  return internEval(input);
}



template <typename element>
arma::vec FunctionTrain<element>::returnGradwrtParameters(arma::vec input)
{
  gradwrtParameters(input);
  return gradwrtParam;
}

template <typename element>
void FunctionTrain<element>::gradwrtParameters(arma::vec input)
{
  whatWeEval=3;

  for (int ii=0;ii<numberOfParameters;ii++)
  {
    updateParametersForGrad(ii,input);
    gradwrtParam(ii)=internEval(input);
  }
}

template <typename element>
void FunctionTrain<element>::updateParametersForGrad(int currentParam,arma::vec input)
{
  int startIndex=0;
  int lastIndex;

  parametersForGrad=parameters;
  for (int ii=0;ii<ninput;ii++)
  {
    // check if currentParam is in the ii matrix
    // evaluate last index of the ii matrix
    lastIndex=startIndex+numberParamElement*ranks(ii)*ranks(ii+1)-1;
    if (currentParam>=startIndex && currentParam <=lastIndex)
      elem.getParametersForGrad(parametersForGrad,currentParam,startIndex,lastIndex,input(ii));
    startIndex=lastIndex+1;
  }
}




template <typename element>
double FunctionTrain<element>::internEval(arma::vec input)
{
  arma::mat valueMatrix=returnInterneMatrix(0,input(0));
  for (int ii=1;ii<ninput;ii++)
  {
    valueMatrix*=returnInterneMatrix(ii,input(ii));
  }

  return arma::as_scalar(valueMatrix);
}



template <typename element>
arma::mat FunctionTrain<element>::returnInterneMatrix(int dimensionNumber,double value)
{
  arma::mat currentMatrix=arma::zeros(ranks(dimensionNumber),ranks(dimensionNumber+1));

  for (int ii=0;ii<ranks(dimensionNumber);ii++)
    for (int jj=0;jj<ranks(dimensionNumber+1);jj++)
    {
      //eval first index
      int baseIndex=evalBaseIndex(dimensionNumber,ii,jj);
      currentMatrix(ii,jj)=returnInterneElement(baseIndex,value);
    }

  return currentMatrix;
}



template <typename element>
int FunctionTrain<element>::evalBaseIndex(int dimensionNumber, int nrow,int ncol)
{
  int baseIndex=0;

  for (int ii=0;ii<dimensionNumber;ii++)
    baseIndex+=ranks(ii)*ranks(ii+1)*numberParamElement;

  baseIndex+=nrow*ranks(dimensionNumber+1)*numberParamElement;
  baseIndex+=ncol*numberParamElement;

  return baseIndex;
}




template <typename element>
double FunctionTrain<element>::returnInterneElement(int firstIndex,double value)
{
  switch (whatWeEval)
  {
    case 1:
      return elem.evalElement(parameters.subvec(firstIndex,firstIndex+numberParamElement-1),value);
      break;
    case 2:
      if (currentDeriveJacobian==currentDimJacobian)
        return elem.evaldElement(parameters.subvec(firstIndex,firstIndex+numberParamElement-1),value);
      else
        return elem.evalElement(parameters.subvec(firstIndex,firstIndex+numberParamElement-1),value);
      break;
    case 3:
      return elem.evalGradwrtParamElement(parametersForGrad.subvec(firstIndex,firstIndex+numberParamElement-1),value);
      break;
  }
}



template <typename element>
void FunctionTrain<element>::update(arma::vec input,double error,double time)
{
  gradwrtParameters(input);
  parameters+=Opti.getUpdateVector(input,error,time,gradwrtParam);
}

#endif
