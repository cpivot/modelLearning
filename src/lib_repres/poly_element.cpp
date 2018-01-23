#include "poly_element.hpp"

polyElement::polyElement()
{

}

polyElement::polyElement(int order_m)
{

  order=order_m;
  numberOfParameters=order+1;

  for (int ii=0;ii<order+1;ii++)
  {
    legendre currentLegendre(ii);
    Poly.push_back(currentLegendre);
    dPoly.push_back(currentLegendre.derive());
  }

}


void polyElement::initialize(arma::vec & parameters,
                              double initialValue,
                              int ninput,
                              arma::vec ranks)
{
  int indexInitialize=0;
  parameters(indexInitialize)=initialValue;
  for (int ii=1;ii<ninput;ii++)
  {
    indexInitialize+=numberOfParameters*ranks(ii-1)*ranks(ii);
    parameters(indexInitialize)=1;
  }
}

int polyElement::returnNumberOfParameters()
{
  return numberOfParameters;
}

double polyElement::evalElement(arma::vec param,double input)
{
  double value=0;
  for (int ii=0;ii<numberOfParameters;ii++)
  value+=param(ii)*Poly.at(ii)(input);
  return value;
}

double polyElement::evaldElement(arma::vec param,double input)
{
  double value=0;
  for (int ii=0;ii<numberOfParameters;ii++)
  value+=param(ii)*dPoly.at(ii)(input);
  return value;
}

void polyElement::getParametersForGrad(arma::vec & parameters,
  int currentParam,
  int startIndex,
  int lastIndex,
  double input)
  {
    for (int ii=startIndex;ii<=lastIndex;ii++)
    parameters(ii)=0.;
    parameters(currentParam)=1.;

  }

  double polyElement::evalGradwrtParamElement(arma::vec param,double input)
  {
    double value=0;
    for (int ii=0;ii<numberOfParameters;ii++)
    value+=param(ii)*Poly.at(ii)(input);
    return value;
  }
