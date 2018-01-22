#include "FT.hpp"

FunctionTrain::FunctionTrain()
{

}

FunctionTrain::FunctionTrain(arma::vec ranks_m,int ninput_m, polyElement poly_m)
{
  ninput=ninput_m;
  ranks=arma::ones(ninput+1);
  ranks.subvec(1,ninput-1)=ranks_m;

  poly=poly_m;
  numberParamElement=poly.returnNumberOfParameters();

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

void FunctionTrain::initialize(double initialValue)
{

    int indexInitialize=0;
    parameters(indexInitialize)=initialValue;
    for (int ii=1;ii<ninput;ii++)
    {
      indexInitialize+=numberParamElement*ranks(ii-1)*ranks(ii);
      parameters(indexInitialize)=1;
    }
}



double FunctionTrain::eval(arma::vec input)
{
  whatWeEval=1;
  return internEval(input);
}

double FunctionTrain::operator()(arma::vec input)
{
  return this->eval(input);
}




arma::vec FunctionTrain::jacobian(arma::vec input)
{
  whatWeEval=2;

  for (int ii=0;ii<ninput;ii++)
  {
    currentDimJacobian=ii;
    jac(ii)=evalElementJacobian(input);
  }

  return jac;
}

double FunctionTrain::evalElementJacobian(arma::vec input)
{
  return internEval(input);
}




arma::vec FunctionTrain::returnGradwrtParameters(arma::vec input)
{
  gradwrtParameters(input);
  return gradwrtParam;
}

void FunctionTrain::gradwrtParameters(arma::vec input)
{
  whatWeEval=3;

  for (int ii=0;ii<numberOfParameters;ii++)
  {
    updateParametersForGrad(ii,input(ii));
    gradwrtParam(ii)=internEval(input);
  }
}

void FunctionTrain::updateParametersForGrad(int currentParam,double value)
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
      poly.getParametersForGrad(parametersForGrad,currentParam,startIndex,lastIndex,value);
    startIndex=lastIndex+1;
  }
}





double FunctionTrain::internEval(arma::vec input)
{
  //fill every matrix
  for (int ii=0;ii<ninput;ii++)
  {
    currentDeriveJacobian=ii;
    interneMatrix.push_back(returnInterneMatrix(ii,input(ii)));
  }

  //compute the matrix product
  arma::mat valueMatrix=interneMatrix.front();
  interneMatrix.pop_front();

  for (int ii=0;ii<ninput-1;ii++)
  {
    valueMatrix*=interneMatrix.front();
    interneMatrix.pop_front();
  }

  return arma::as_scalar(valueMatrix);
}




arma::mat FunctionTrain::returnInterneMatrix(int dimensionNumber,double value)
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




int FunctionTrain::evalBaseIndex(int dimensionNumber, int nrow,int ncol)
{
  int baseIndex=0;

  for (int ii=0;ii<dimensionNumber;ii++)
    baseIndex+=ranks(ii)*ranks(ii+1)*numberParamElement;

  baseIndex+=nrow*ranks(dimensionNumber+1)*numberParamElement;
  baseIndex+=ncol*numberParamElement;

  return baseIndex;
}





double FunctionTrain::returnInterneElement(int firstIndex,double value)
{
  if (whatWeEval==1)
    return poly.evalElement(parameters.subvec(firstIndex,firstIndex+numberParamElement-1),value);

  if (whatWeEval==2)
  {
    if (currentDeriveJacobian==currentDimJacobian)
      return poly.evaldElement(parameters.subvec(firstIndex,firstIndex+numberParamElement-1),value);
    else
      return poly.evalElement(parameters.subvec(firstIndex,firstIndex+numberParamElement-1),value);
  }

  if (whatWeEval==3)
    return poly.evalGradwrtParamElement(parametersForGrad.subvec(firstIndex,firstIndex+numberParamElement-1),value);

}




void FunctionTrain::update(arma::vec input,double error,double time)
{
  gradwrtParameters(input);
  parameters+=Opti.getUpdateVector(input,error,time,gradwrtParam);
}
