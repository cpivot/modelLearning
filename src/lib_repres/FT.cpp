#include "FT.hpp"





template <typename element>
FunctionTrain<element>::FunctionTrain()
{

}

template <typename element>
void FunctionTrain<element>::define(arma::vec ranks_m,int ninput_m, element elem_m)
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

  initialize(1);

}

template <typename element>
void FunctionTrain<element>::initialize(double initialValue)
{
  elem.initialize(parameters,initialValue,ninput,ranks);
}

template <typename element>
int FunctionTrain<element>::returnNumberOfParameters()
{
  return numberOfParameters;
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
void FunctionTrain<element>::randomize()
{
  parameters.randn();
  parameters*=0.1;
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
    lastIndex=startIndex+numberParamElement*ranks(ii)*ranks(ii+1)-1;
    if (currentParam>=startIndex && currentParam<=lastIndex)
      elem.getParametersForGrad(parametersForGrad,currentParam,startIndex,lastIndex,input(ii));
    startIndex=lastIndex+1;
  }
}




template <typename element>
double FunctionTrain<element>::internEval(arma::vec input)
{
  int baseIndex=0;
  arma::mat valueMatrix=returnInterneMatrix(0,input(0),baseIndex);
  for (int ii=1;ii<ninput;ii++)
    valueMatrix*=returnInterneMatrix(ii,input(ii),baseIndex);

  return arma::as_scalar(valueMatrix);
}



template <typename element>
arma::mat FunctionTrain<element>::returnInterneMatrix(int dimensionNumber,double value,int & baseIndex)
{
  arma::mat currentMatrix=arma::zeros(ranks(dimensionNumber),ranks(dimensionNumber+1));
  for (int ii=0;ii<ranks(dimensionNumber);ii++)
    for (int jj=0;jj<ranks(dimensionNumber+1);jj++)
    {
      currentMatrix(ii,jj)=returnInterneElement(baseIndex,value);
      baseIndex+=numberParamElement;
    }

  return currentMatrix;
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
void FunctionTrain<element>::updateParameters(arma::vec parrametersCorrection)
{
  parameters+=parrametersCorrection;
//  cout << arma::norm(parrametersCorrection) << endl;
}



// Explicit template instantiation
template class FunctionTrain<polyElement>;
template class FunctionTrain<kernelElement>;
