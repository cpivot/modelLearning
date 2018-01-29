#include "adadelta.hpp"

Adadelta::Adadelta()
{

}

void Adadelta::define(int nparam_m,double signe_m,double rho_m,double nonuseA,double nouseB)
{

  nparam=nparam_m;
  rho=rho_m;

  Eg=arma::zeros(nparam);
  Edx=arma::zeros(nparam);
  dx=arma::zeros(nparam);

  signe=signe_m;
}

arma::vec Adadelta::RMS(arma::vec input)
{
  return sqrt(input+1e-6);
}

arma::vec Adadelta::getUpdateVector(arma::vec X, double err,arma::vec nablaParams,double t)
{

  nablaParams=err*nablaParams;
  Eg=rho*Eg+(1.-rho)*square(nablaParams);
  dx=signe*RMS(Edx)/RMS(Eg)%nablaParams;
  Edx=rho*Edx+(1.-rho)*square(dx);

  return dx;
}
