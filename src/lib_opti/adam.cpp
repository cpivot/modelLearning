#include "adam.hpp"


Adam::Adam(int nparam_m,double alpha_m,double betaun_m,double betadeux_m,double signe_m)
{

  nparam=nparam_m;
  betaun=betaun_m;
  betadeux=betadeux_m;
  alpha=alpha_m;
  signe=signe_m;

  updateVector=arma::zeros(nparam);

  m=arma::zeros(nparam);
  v=arma::zeros(nparam);

  mBias=arma::zeros(nparam);
  vBias=arma::zeros(nparam);
}


arma::vec Adam::getUpdateVector(arma::vec X, double err,double t,arma::vec nablaParams)
{
  nablaParams= err*nablaParams;

  m=betaun*m+(1-betaun)*nablaParams;
  v=betadeux*v+(1-betadeux)*arma::square(nablaParams);

  mBias=m/(1-std::pow(betaun,t));
  vBias=v/(1-std::pow(betadeux,t));

  for (int jj=0; jj < nparam;jj++)
    updateVector(jj)=mBias(jj)/(sqrt(vBias(jj))+1e-8);

  return signe*alpha*updateVector;
}
