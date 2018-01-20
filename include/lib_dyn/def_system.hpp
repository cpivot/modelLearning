#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/progress.hpp>

#include <stdlib.h>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace boost::numeric::odeint;


static double controlR=0;
static double controlRa=0;
static double controlRb=0;



//**************************
// Rossler
//**************************

// Rossler's Parameters
typedef boost::array< double , 3 > state_type_rossler;

const double aR=0.432;
const double bR=2.0;
const double cR=4.0;

// Define the Rossler's System
void rossler(const state_type_rossler &x , state_type_rossler &dxdt , double t )
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
void lorenz(const state_type_lorenz &x , state_type_lorenz &dxdt , double t )
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
const double EsigmaR=0.5;
const double ErhoR=14.22;
const double EaR=.5;
const double EaRq=aR*aR;
const double Eb[]={4.*(1.+EaRq)/(1.+2.*EaRq),
                  (1.+2.*EaRq)/(2.*(1.+EaRq)),
                  2*(1.-EaRq)/(1.+EaRq),
                  EaRq/(1.+EaRq),
                  8.*EaRq/(1.+2.*EaRq),
                  4./(1.+2.*EaRq)};


// Define the Rossler's System
void lorenznined(const state_type_extended_lorenz &x , state_type_extended_lorenz &dxdt , double t )
{
  dxdt[0] = -EsigmaR*Eb[0]*x[0]-x[1]*x[3]+Eb[3]*x[3]*x[3]+Eb[2]*x[2]*x[4]-EsigmaR*Eb[1]*x[6];
  dxdt[1] = -EsigmaR*x[1]+x[0]*x[3]-x[1]*x[4]+x[3]*x[4]-.5*EsigmaR*x[8];
  dxdt[2] = -EsigmaR*Eb[0]*x[2]+x[1]*x[3]-Eb[3]*x[1]*x[1]-Eb[2]*x[0]*x[4]+EsigmaR*Eb[1]*x[7]+controlR;
  dxdt[3] = -EsigmaR*x[3]-x[1]*x[2]-x[1]*x[4]+x[3]*x[4]+.5*EsigmaR*x[8];
  dxdt[4] = -EsigmaR*Eb[4]*x[4]+.5*x[1]*x[1]-.5*x[3]*x[3];
  dxdt[5] = -Eb[5]*x[5]+x[1]*x[8]-x[3]*x[8];
  dxdt[6] = -Eb[0]*x[6]-ErhoR*x[0]+2*x[4]*x[7]-x[3]*x[8];
  dxdt[7] = -Eb[0]*x[7]+ErhoR*x[2]-2*x[4]*x[6]+x[1]*x[8];
  dxdt[8] = -x[8]-ErhoR*x[1]+ErhoR*x[3]-2*x[1]*x[5]+2*x[3]*x[5]+x[3]*x[6]-x[1]*x[7];
}
