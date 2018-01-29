#include "model.hpp"



template <typename representation, typename optimizer>
model<representation,optimizer>::model()
{

}

template <typename representation, typename optimizer>
model<representation,optimizer>::model(representation repres_m,optimizer opti_m,arma::mat bound_m)
{
  repres=repres_m;
  opti=opti_m;
  Bound=bound_m;
  numberOfParameters=repres.returnNumberOfParameters();
}

template <typename representation, typename optimizer>
void model<representation,optimizer>::define(representation repres_m,optimizer opti_m,arma::mat bound_m)
{
  repres=repres_m;
  opti=opti_m;
  Bound=bound_m;
}

template <typename representation, typename optimizer>
void model<representation,optimizer>::randomizeParameters()
{
  repres.randomize();
}

template <typename representation, typename optimizer>
double model<representation,optimizer>::eval(arma::vec input)
{
  //wrap input to bound
  arma::vec wrapInput=arma::zeros(input.n_elem);
  for (int ii=0;ii<input.n_elem;ii++)
    wrapInput(ii)=0.5*input(ii)/(Bound(ii,0)-Bound(ii,1))-(Bound(ii,0)+Bound(ii,1))/(Bound(ii,0)-Bound(ii,1));

  return repres(wrapInput);
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
  time=t;
  arma::vec gradwrtParams=repres.returnGradwrtParameters(X);
  //cout << arma::norm(gradwrtParams) << endl;
  gradwrtParams%=1.+0.1*arma::randn(gradwrtParams.size())/(std::pow(1.+time,0.55));
  //cout << arma::norm(gradwrtParams) << endl;
  //cout << err << endl;
  //cout << "****" << endl;
  arma::vec updateParamsVec=opti.getUpdateVector(X,err,gradwrtParams,t);
  repres.updateParameters(updateParamsVec);
}


// Explicit template instantiation
template class model<FunctionTrain<kernelElement>,Adam>;
template class model<FunctionTrain<polyElement>,Adam>;

template class model<FunctionTrain<kernelElement>,Adadelta>;
template class model<FunctionTrain<polyElement>,Adadelta>;
