#include "FT.hpp"

FunctionTrain::FunctionTrain(arma::vec ranks_m,int ninput_m, int order_m, double initialValue)
{
  ninput=ninput_m;
  order=order_m+1;
  ranks=arma::ones(ninput+1);
  ranks.subvec(1,ninput-1)=ranks_m;
  //ranks.print();

  jac=arma::zeros(ninput);

  for (int ii=0;ii<order;ii++)
  {
    legendre currentLegendre(ii);
    PolyLegendre.push_back(currentLegendre);
    dPolyLegendre.push_back(currentLegendre.derive());
  }

  //Evaluate number of parameters
  numberOfParameters=0;
  for (int ii=0;ii<ninput;ii++)
    numberOfParameters+=ranks(ii)*ranks(ii+1);

  numberOfParameters*=order;
  parameters=arma::zeros(numberOfParameters);

  parametersForGrad=arma::zeros(numberOfParameters);
  gradwrtParam=arma::zeros(numberOfParameters);

  initialize(initialValue);
}

void FunctionTrain::initialize(double initialValue)
{

    int indexInitialize=0;
    parameters(indexInitialize)=initialValue;
    for (int ii=1;ii<ninput;ii++)
    {
      indexInitialize+=order*ranks(ii-1)*ranks(ii);
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





arma::vec FunctionTrain::gradwrtParameters(arma::vec input)
{
  whatWeEval=3;

  for (int ii=0;ii<numberOfParameters;ii++)
  {
    updateParametersForGrad(ii);
    gradwrtParam(ii)=internEval(input);
  }

  return gradwrtParam;
}


void FunctionTrain::updateParametersForGrad(int currentParam)
{
  int startIndex=0;
  int lastIndex;

  parametersForGrad=parameters;

  for (int ii=0;ii<ninput;ii++)
  {
    // check if currentParam is in the ii matrix
    // evaluate last index of the ii matrix
    lastIndex=order*ranks(ii)*ranks(ii+1)-1;
    if (currentParam>=startIndex && currentParam <=lastIndex)
    {
      for (ii=startIndex;ii<=lastIndex;ii++)
        parametersForGrad(ii)=0;
      parametersForGrad(currentParam)=1;
    }
    startIndex=lastIndex+1;
  }
}






double FunctionTrain::internEval(arma::vec input)
{
  //fill every matrix
  for (int ii=0;ii<ninput;ii++)
    interneMatrix.push_back(returnInterneMatrix(ii,input(ii)));

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
    baseIndex+=ranks(ii)*ranks(ii+1)*order;

  baseIndex+=nrow*ranks(dimensionNumber+1)*order;
  baseIndex+=ncol*order;

  return baseIndex;
}

double FunctionTrain::returnInterneElement(int firstIndex,double value)
{
  double currentValue=0.;

  if (whatWeEval==1)
    for (int ii=0;ii<order;ii++)
      currentValue+=parameters(firstIndex+ii)*PolyLegendre.front()(value);

  if (whatWeEval==2)
  {
    for (int ii=0;ii<order;ii++)
      if (ii==currentDimJacobian)
        currentValue+=parameters(firstIndex+ii)*dPolyLegendre.front()(value);
      else
        currentValue+=parameters(firstIndex+ii)*PolyLegendre.front()(value);
  }

  if (whatWeEval==3)
    for (int ii=0;ii<order;ii++)
      currentValue+=parametersForGrad(firstIndex+ii)*PolyLegendre.front()(value);

  return currentValue;
}
