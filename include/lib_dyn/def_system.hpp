#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/progress.hpp>

#include <stdlib.h>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <algorithm>
#include <numeric>

#include "reinforcement.hpp"

using namespace std;
using namespace boost::numeric::odeint;


static double controlR=0;


typedef boost::array< double , 3 > state_type_rossler;


//**************************
// Rossler
//**************************

// Rossler's Parameters
const double aR=0.432;
const double bR=2.0;
const double cR=4.0;

// Define the Rossler's System
void rossler(const state_type &x , state_typerossler &dxdt , double t )
{
  dxdt[0] = -x[1]-x[2];
  dxdt[1] = x[0]+aR*x[1]+controlR;
  dxdt[2] = bR+x[2]*(x[0]-cR);
}


//**************************
// Lorenz
//**************************


typedef boost::array< double , 3 > state_type_lorenz;

// Rossler's Parameters
const double sigmaR=10.;
const double rhoR=28.;
const double betaR=8./3.;

// Define the Rossler's System
void lorenz(const state_type &x , state_type_lorenz &dxdt , double t )
{
  dxdt[0] = sigmaR*(x[1]-x[0]);
  dxdt[1] = x[0]*(rhoR-x[2])-x[1]+controlR;
  dxdt[2] = x[0]*x[1]-betaR*x[2];
}



//**************************
// Lorenz 9D
//**************************

typedef boost::array< double , 9 > state_type_extended_lorenz;

// Rossler's Parameters
const double sigmaR=0.5;
const double rhoR=14.22;
const double aR=.5;
const double aRq=aR*aR;
const double b[]={4.*(1.+aRq)/(1.+2.*aRq),
                  (1.+2.*aRq)/(2.*(1.+aRq)),
                  2*(1.-aRq)/(1.+aRq),
                  aRq/(1.+aRq),
                  8.*aRq/(1.+2.*aRq),
                  4./(1.+2.*aRq)};


// Define the Rossler's System
void lorenznined(const state_type &x , state_type &dxdt , double t )
{
  dxdt[0] = -sigmaR*b[0]*x[0]-x[1]*x[3]+b[3]*x[3]*x[3]+b[2]*x[2]*x[4]-sigmaR*b[1]*x[6];
  dxdt[1] = -sigmaR*x[1]+x[0]*x[3]-x[1]*x[4]+x[3]*x[4]-.5*sigmaR*x[8];
  dxdt[2] = -sigmaR*b[0]*x[2]+x[1]*x[3]-b[3]*x[1]*x[1]-b[2]*x[0]*x[4]+sigmaR*b[1]*x[7]+controlR;
  dxdt[3] = -sigmaR*x[3]-x[1]*x[2]-x[1]*x[4]+x[3]*x[4]+.5*sigmaR*x[8];
  dxdt[4] = -sigmaR*b[4]*x[4]+.5*x[1]*x[1]-.5*x[3]*x[3];
  dxdt[5] = -b[5]*x[5]+x[1]*x[8]-x[3]*x[8];
  dxdt[6] = -b[0]*x[6]-rhoR*x[0]+2*x[4]*x[7]-x[3]*x[8];
  dxdt[7] = -b[0]*x[7]+rhoR*x[2]-2*x[4]*x[6]+x[1]*x[8];
  dxdt[8] = -x[8]-rhoR*x[1]+rhoR*x[3]-2*x[1]*x[5]+2*x[3]*x[5]+x[3]*x[6]-x[1]*x[7];
}
