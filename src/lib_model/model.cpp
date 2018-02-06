#include "model.hpp"



template <typename representation, typename optimizer>
model<representation,optimizer>::model()
{

}




template <typename representation, typename optimizer>
model<representation,optimizer>::model(string file)
{
  arma::arma_rng::set_seed_random();

  parametersFile=file;

  pt::ptree root;
  pt::read_json(parametersFile.c_str(), root);

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
    repres.randomize();




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


  betaExplo=root.get<double>("Model.Exploration.beta", 0.1);
  rhoExplo=root.get<double>("Model.Exploration.rho", 0.5);
  explo.define(ranks,ninput,elem);
  explo.defineElement(order);
  explo.evaluateNumberOfParameters();
  int numberOfParametersExplo=explo.returnNumberOfParameters();
  explo.initialize(0.001);

  doMonitoring=root.get<bool>("Model.Monitoring.save", true);

}






template <typename representation, typename optimizer>
model<representation,optimizer>::~model()
{
  pt::ptree root;
  pt::read_json(parametersFile.c_str(), root);

  string fileMonitoring=root.get<string>("Model.Monitoring.file","default_Monitoring.dat");

  if (doMonitoring)
  {
    ofstream save;
    save.open(fileMonitoring.c_str());
    monitoringModel current;

    save << std::setw(10) << "time" << "\t" << "Error" << "\t" << "Coef Explo" << endl;

// Bug bug bug!!!!
// WHY monitoring.size()/2+1 instead of monitoring.size()
    arma::vec forHistogram=arma::zeros(monitoring.size()/2+1);

    for (int ii=0;ii<monitoring.size();ii++)
    {
      current=monitoring.front();
      save << std::setw(10) << current.time << "\t";
      save << std::setw(10) << current.error << "\t";
      save << std::setw(10) << current.coefExplo << "\t";
      save << std::endl;
      forHistogram(ii)=std::abs(current.error);
      monitoring.pop_front();
    }

    save.close();

    bool doHistrogram=root.get<bool>("Model.Monitoring.doHistrogram", true);
    if (doHistrogram)
    {
      vec a = arma::logspace<arma::vec>(-4, 3,30);
      uvec h1 = hist(forHistogram(arma::span(std::floor(forHistogram.n_elem*6/7),forHistogram.n_elem-1)), a);
      arma::mat totalHist=arma::zeros(30,2);
      totalHist.col(0)=arma::conv_to<colvec>::from(a);
      totalHist.col(1)=arma::conv_to<colvec>::from(h1);
      totalHist.save("histogram.dat",arma::raw_ascii);
    }

    save.open("representation.dat");
    for (int ii=0;ii<saveRepres.size();ii++)
    {
      arma::vec currentX=saveRepres.front();
      for (int jj=0;jj<ninput;jj++)
        save << std::setw(10) << currentX(jj) << "\t";
      wrapInput(currentX);
      save << std::setw(10) << repres(currentX) << "\t";
      save << std::setw(10) << explo(currentX) << "\t";

      save << std::endl;
      saveRepres.pop_front();
    }
    save.close();

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
void model<representation,optimizer>::wrapInput(arma::vec & input)
{
  //wrap input to bound
  for (int ii=0;ii<input.n_elem;ii++)
    input(ii)=-2*input(ii)/(Bound(ii,1)-Bound(ii,0))+(Bound(ii,1)+Bound(ii,0))/(Bound(ii,1)-Bound(ii,0));
}





template <typename representation, typename optimizer>
void model<representation,optimizer>::addExploration(arma::vec input)
{
//  input.print("input before");
  wrapInput(input);
//  input.print("input after");
  double currentExplor=explo(input);
  arma::vec gradwrtParams=explo.returnGradwrtParameters(input);
//  cout << arma::norm(gradwrtParams) << endl;
  explo.updateParameters(1e-5*gradwrtParams);
  explo.sat(1e6);

  arma::vec randomVector=arma::randn(numberOfParameters);
  coefExplo=1+std::abs(explo(input));
//  cout << coefExplo << endl;

  repres.updateParameters(randomVector*betaExplo/std::pow(coefExplo,rhoExplo));
}






template <typename representation, typename optimizer>
double model<representation,optimizer>::eval(arma::vec input)
{
  wrapInput(input);
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
  wrapInput(input);
  return repres.jacobian(input);
}







template <typename representation, typename optimizer>
double model<representation,optimizer>::update(arma::vec X, double Xnext,double t)
{
  arma::vec Xsave=X;
  wrapInput(X);

  time=t;
  double currentLearningRate;

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

  addExploration(X);


  if (doMonitoring)
  {
    currentMonitoring.time=time;
    currentMonitoring.error=error;
    currentMonitoring.coefExplo=coefExplo;
    monitoring.push_back(currentMonitoring);
    arma::vec vecReps=arma::zeros(ninput);
    vecReps(arma::span(0,ninput-1))=Xsave;
    saveRepres.push_back(vecReps);
  }

  return error;
}





// Explicit template instantiation
template class model<FunctionTrain<kernelElement>,Adam>;
template class model<FunctionTrain<polyElement>,Adam>;

template class model<FunctionTrain<kernelElement>,Adadelta>;
template class model<FunctionTrain<polyElement>,Adadelta>;
