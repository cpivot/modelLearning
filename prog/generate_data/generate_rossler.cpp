#include <iostream>

#include <armadillo>
#include <math.h>

#include "def_system.hpp"

#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/progress.hpp>


int main(int argc, char** argv)
{
  int ninput=3;
  int Ntest=1e5;
  double dt=1e-1;

  state_type_rossler x={{1.0,1.0,1.0}};
  integrate( rossler , x , 0.0 , 200.0 , 0.1/3);

  arma::vec X=arma::zeros(ninput);
  arma::mat total=arma::zeros(Ntest,1+ninput);

  for (int ii=1;ii<Ntest;ii++)
  {
    integrate( rossler , x , 0.0 , dt , dt/2);

    for (int kk=0;kk<ninput;kk++)
      X(kk)=x[kk];

    total(ii,0)=dt*ii;
    total(ii,arma::span(1,ninput))=X.t();
  }

    total.save("rossler.dat",arma::raw_ascii);

  return 0;
}
