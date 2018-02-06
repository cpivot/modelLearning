#include <iostream>
#include <armadillo>
#include <math.h>

#include "lib_opti.hpp"
#include "lib_base.hpp"
#include "lib_repres.hpp"

#include "loss_function.hpp"

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


#ifndef ____MODEL__
#define ____MODEL__


struct monitoringModel
{
  double time;
  double error;
  double coefExplo;
};

namespace pt = boost::property_tree;


template <typename representation, typename optimizer>
class model
{
private:
  string parametersFile;

  representation repres;
  optimizer opti;
  double time; // Global value for time
  int lossFunction;

  representation explo;
  Adadelta optiExplo;
  double betaExplo;
  double rhoExplo;
  double coefExplo;

  int numberOfParameters;

  arma::mat Bound;
  int ninput;

  bool doMonitoring;
  std::list<monitoringModel> monitoring;
  monitoringModel currentMonitoring;
  std::list<arma::vec> saveRepres;

public:
  model();
  model(string);
  ~model();

  void define(representation,optimizer,arma::mat);
  void randomizeParameters();
  void addExploration(arma::vec);

  int returnnNinput();

  void wrapInput(arma::vec &);
  double eval(arma::vec);
  double operator()(arma::vec);

  arma::vec jacobian(arma::vec);
  double update(arma::vec,double,double t=1);

};




#endif
