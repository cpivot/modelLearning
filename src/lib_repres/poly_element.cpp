#include "poly_element.hpp"

polyElement::polyElement()
{

}

void polyElement::define(int order_m,bool checkB)
{

  order=order_m;
  numberOfParameters=order+1;

  for (int ii=0;ii<order+1;ii++)
  {
    legendre currentLegendre(ii,checkB);
    Poly.push_back(currentLegendre.returnCoef());
    dPoly.push_back(currentLegendre.derive().returnCoef());
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



double polyElement::eval(arma::vec coef,double x)
{
  double result=coef(0);
  for (int ii=1;ii<coef.n_elem;ii++)
    result+=coef(ii)*std::pow(x,ii);

  return result;
}


double polyElement::evalElement(arma::vec param,double input)
{
  double value=0;
  for (int ii=0;ii<numberOfParameters;ii++)
    value+=param(ii)*eval(Poly.at(ii),input);
  return value;
}

double polyElement::evaldElement(arma::vec param,double input)
{
  double value=0;
  for (int ii=0;ii<numberOfParameters;ii++)
    value+=param(ii)*eval(dPoly.at(ii),input);
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
  {
    value+=param(ii)*eval(Poly.at(ii),input);
  }

  return value;
}
