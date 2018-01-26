#include "model.hpp"



template <typename representation, typename optimizer>
model<representation,optimizer>::model()
{

}

template <typename representation, typename optimizer>
model<representation,optimizer>::model(representation repres_m,optimizer opti_m)
{
  repres=repres_m;
  opti=opti_m;
}

template <typename representation, typename optimizer>
void model<representation,optimizer>::define(representation repres_m,optimizer opti_m)
{
  repres=repres_m;
  opti=opti_m;
}

template <typename representation, typename optimizer>
void model<representation,optimizer>::randomizeParameters()
{
  repres.randomize();
}

template <typename representation, typename optimizer>
double model<representation,optimizer>::eval(arma::vec input)
{
  return repres(input);
}

template <typename representation, typename optimizer>
double model<representation,optimizer>::operator()(arma::vec input)
{
  return this->eval(input);
}

template <typename representation, typename optimizer>
arma::vec model<representation,optimizer>::jacobian(arma::vec input)
{
  return repres.jacobian(input);
}

template <typename representation, typename optimizer>
void model<representation,optimizer>::update(arma::vec X, double err,double t)
{
  arma::vec gradwrtParams=repres.returnGradwrtParameters(X);
  arma::vec updateParamsVec=opti.getUpdateVector(X,err,gradwrtParams,t);
  repres.updateParameters(updateParamsVec);
}


// Explicit template instantiation
template class model<FunctionTrain<kernelElement>,Adam>;
template class model<FunctionTrain<polyElement>,Adam>;

template class model<FunctionTrain<kernelElement>,Adadelta>;
template class model<FunctionTrain<polyElement>,Adadelta>;
