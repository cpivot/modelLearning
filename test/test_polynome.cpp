#include <iostream>

#include <armadillo>
#include <math.h>

#include "polynome.hpp"

int main(int argc, char** argv)
{
  int order=4;

  std::cout << "Define test" << std::endl;
  polynom test;
  test.define(order);
  test.affiche();

  arma::vec coef=arma::regspace(0,order);
  test.define(coef);
  test.affiche();
  test.updateCoef(3,10.);
  test.affiche();
  std::cout << test(2.) << std::endl;

  std::cout << "Derive test" << std::endl;
  polynom derive_test=test.derive();
  derive_test.affiche();

  test.d();
  test.affiche();


  std::cout << "Addition test" << std::endl;
  polynom testb;
  arma::vec coefb=arma::regspace(0,order-2);
  testb.define(coefb);
  testb.affiche();

  polynom testc=test+testb;
  testc.affiche();

  polynom testd;
  arma::vec coefd=arma::regspace(0,order+1);
  testd.define(coefd);
  testd.affiche();

  polynom teste=test+testd;
  teste.affiche();

  std::cout << "Soutraction test" << std::endl;
  polynom testg=test-testb;
  testg.affiche();

  std::cout << "Multiplication test" << std::endl;
  polynom testf=test*testc;
  testf.affiche();



  std::cout << "Evaluation test" << std::endl;
  std::cout << testf(3.0) << std::endl;

  /*
  std::cout << std::endl << std::endl;
  std::cout << "Legendre" << std::endl;

  for (int ii=0;ii<8;ii++)
    {
      legendre PolyLegendre(ii);
      PolyLegendre.affiche();
    }

  legendre Polylegendre(4);
  std::cout << Polylegendre(0.5) << std::endl;

  std::cout << Polylegendre(-2.) << std::endl;

  std::cout << "derive Legendre" << std::endl;
  for (int ii=0;ii<8;ii++)
    {
      std::cout << "Order Legendre:" << ii << std::endl;
      legendre PolyLegendre(ii);
      polynom derivePolyLegendre=PolyLegendre.derive();
      derivePolyLegendre.affiche();
      std::cout << derivePolyLegendre(2.) << std::endl;
    }
  */

  return 0;
}
