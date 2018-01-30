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

  int ninput=root.get<int>("Dimension", 0);
  int order=root.get<int>("Representation.FunctionTrain.polyElement.order", 0);

  arma::vec ranks=arma::zeros(ninput-1);
  int ii=0;
  BOOST_FOREACH(pt::ptree::value_type &v, root.get_child("Representation.FunctionTrain.Ranks"))
  {
      ranks(ii)=v.second.get_value<double>();
      ii++;
  }

  Bound=arma::zeros(ninput,2);
  ii=0;
  BOOST_FOREACH(pt::ptree::value_type &v, root.get_child("LowerBounds"))
  {
      Bound(ii,1)=v.second.get_value<double>();
      ii++;
  }

  ii=0;
  BOOST_FOREACH(pt::ptree::value_type &v, root.get_child("UpperBounds"))
  {
      Bound(ii,0)=v.second.get_value<double>();
      ii++;
  }

  typename representation::elementType elem;
  repres.define(ranks,ninput,elem);

  repres.defineElement(order);

  repres.evaluateNumberOfParameters();
  numberOfParameters=repres.returnNumberOfParameters();

  double initialValue=root.get<double>("Representation.initialValue", 0.5);
  repres.initialize(initialValue);

  if (root.get<bool>("Representation.randomize", false))
    randomizeParameters();




  int signeOpti=root.get<int>("Optimizer.signe", 1);
  if (typeid(optimizer).name()==typeid(Adam).name())
  {
    double alpha=root.get<double>("Optimizer.Adam.alpha", 0.001);
    double beta_un=root.get<double>("Optimizer.Adam.beta_un", 0.9);
    double beta_deux=root.get<double>("Optimizer.Adam.beta_deux", 0.999);

    opti.define(numberOfParameters,alpha,signeOpti,beta_un,beta_deux);
  }
  if (typeid(optimizer).name()==typeid(Adadelta).name())
  {
    double rho=root.get<double>("Optimizer.Adadelta.rho", 0.95);
    opti.define(numberOfParameters,signeOpti,rho);
  }

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
void model<representation,optimizer>::addExploration()
{
  arma::vec randomVector=arma::randn(numberOfParameters);
  randomVector*=0.0*(std::pow(1.+time,0.55));
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
void model<representation,optimizer>::update(arma::vec X, double err,double t)
{
  time=t;
  arma::vec gradwrtParams=repres.returnGradwrtParameters(X);
  //cout << arma::norm(gradwrtParams) << endl;
//  gradwrtParams%=1.+0.5*arma::randn(gradwrtParams.size())/(std::pow(1.+time,0.55));
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
