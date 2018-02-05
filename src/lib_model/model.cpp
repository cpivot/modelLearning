#include "model.hpp"



template <typename representation, typename optimizer>
model<representation,optimizer>::model()
{

}

template <typename representation, typename optimizer>
model<representation,optimizer>::model(string file)
{
  pt::ptree root;
  pt::read_json(file.c_str(), root);

  ninput=root.get<int>("Model.Dimension", 0);
  int order=root.get<int>("Model.Representation.FunctionTrain.polyElement.order", 0);

  arma::vec ranks=arma::zeros(ninput-1);
  int ii=0;
  BOOST_FOREACH(pt::ptree::value_type &v, root.get_child("Model.Representation.FunctionTrain.Ranks"))
  {
      ranks(ii)=v.second.get_value<double>();
      ii++;
  }

  Bound=arma::zeros(ninput,2);
  ii=0;
  BOOST_FOREACH(pt::ptree::value_type &v, root.get_child("Model.LowerBounds"))
  {
      Bound(ii,1)=v.second.get_value<double>();
      ii++;
  }

  ii=0;
  BOOST_FOREACH(pt::ptree::value_type &v, root.get_child("Model.UpperBounds"))
  {
      Bound(ii,0)=v.second.get_value<double>();
      ii++;
  }

  typename representation::elementType elem;
  repres.define(ranks,ninput,elem);

  repres.defineElement(order);

  repres.evaluateNumberOfParameters();
  numberOfParameters=repres.returnNumberOfParameters();

  double initialValue=root.get<double>("Model.Representation.initialValue", 0.5);
  repres.initialize(initialValue);

  if (root.get<bool>("Model.Representation.randomize", false))
    randomizeParameters();




  int signeOpti=root.get<int>("Model.Optimizer.signe", 1);
  if (typeid(optimizer).name()==typeid(Adam).name())
  {
    double alpha=root.get<double>("Model.Optimizer.Adam.alpha", 0.001);
    double beta_un=root.get<double>("Model.Optimizer.Adam.beta_un", 0.9);
    double beta_deux=root.get<double>("Model.Optimizer.Adam.beta_deux", 0.999);

    opti.define(numberOfParameters,alpha,signeOpti,beta_un,beta_deux);
  }
  if (typeid(optimizer).name()==typeid(Adadelta).name())
  {
    double rho=root.get<double>("Model.Optimizer.Adadelta.rho", 0.95);
    opti.define(numberOfParameters,signeOpti,rho);
  }

  string lossFunctionString=root.get<string>("Model.Optimizer.LossFunction","phiHL");
  if (lossFunctionString=="phiGM")
    lossFunction=1;
  if (lossFunctionString=="phiHL")
    lossFunction=2;
  if (lossFunctionString=="phiHS")
    lossFunction=3;
  if (lossFunctionString=="phiGR")
    lossFunction=4;

}






template <typename representation, typename optimizer>
void model<representation,optimizer>::define(representation repres_m,optimizer opti_m,arma::mat bound_m)
{
  repres=repres_m;
  opti=opti_m;
  Bound=bound_m;
}


template <typename representation, typename optimizer>
int model<representation,optimizer>::returnnNinput()
{
  return ninput;
}


template <typename representation, typename optimizer>
void model<representation,optimizer>::randomizeParameters()
{
  repres.randomize();
}

template <typename representation, typename optimizer>
void model<representation,optimizer>::addExploration()
{
  arma::vec randomVector=arma::randn(numberOfParameters);
  randomVector*=.1*(std::pow(1.+time,0.1));
  repres.updateParameters(randomVector);
}

template <typename representation, typename optimizer>
double model<representation,optimizer>::eval(arma::vec input)
{
  //wrap input to bound
  arma::vec wrapInput=arma::zeros(input.n_elem);
  for (int ii=0;ii<input.n_elem;ii++)
    wrapInput(ii)=-2*input(ii)/(Bound(ii,1)-Bound(ii,0))+(Bound(ii,1)+Bound(ii,0))/(Bound(ii,1)-Bound(ii,0));

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
void model<representation,optimizer>::update(arma::vec X, double Xnext,double t)
{
  time=t;
  double evalModel=repres(X);
  arma::vec gradwrtParams=repres.returnGradwrtParameters(X);

  double error=evalModel-Xnext;
  double errorValue;
  double errorDerivate;

  switch (lossFunction)
  {
    case 1:
    phiGM(error,errorValue,errorDerivate);
    break;
    case 2:
    phiHL(error,errorValue,errorDerivate);
    break;
    case 3:
    phiHS(error,errorValue,errorDerivate);
    break;
    case 4:
    phiGR(error,errorValue,errorDerivate);
    break;
  }

  arma::vec updateParamsVec=opti.getUpdateVector(X,errorDerivate,gradwrtParams,t);
  repres.updateParameters(updateParamsVec);
}





// Explicit template instantiation
template class model<FunctionTrain<kernelElement>,Adam>;
template class model<FunctionTrain<polyElement>,Adam>;

template class model<FunctionTrain<kernelElement>,Adadelta>;
template class model<FunctionTrain<polyElement>,Adadelta>;
