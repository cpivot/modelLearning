#ifndef ____polynom__
#define ____polynom__

#include <iostream>
#include <armadillo>
#include <math.h>


using namespace std;
using namespace arma;

class polynom
{
protected:
  int order;
  vec coef;

public:
  polynom();
  ~polynom();
  void define(vec);
  void define(int);

  void reset();
  void affiche();
  double eval(double);
  double operator()(double);

  vec returnCoef();
  void updateCoef(int, double);
  void updateCoef(vec);

  int returnOrder();
  void updateOrder(int);

  polynom derive(); // return the derivate polynom
  void d(); // derive the current polynome
  polynom integrate(); // return the integrate polynom
  //  void integrate(); //integrate the current polynom


  void add(polynom);
  void add(double);
  void multiply(polynom);
  void multiply(double);

  //operator overloading


  void operator=(polynom const&);
  void operator+=(polynom const&);
  void operator+=(double const&);
  void operator+=(int const&);
  void operator*=(polynom const&);
  void operator*=(double const&);
  void operator*=(int const&);
};


polynom operator+(polynom const&, polynom const&);
polynom operator+(polynom const&, double const&);
polynom operator+(double const&,polynom const&);
polynom operator+(polynom const&, int const&);
polynom operator+(int const&,polynom const&);

polynom operator*(polynom const&, polynom const&);
polynom operator*(polynom const&, double const&);
polynom operator*(double const&, polynom const&);
polynom operator*(polynom const&, int const&);
polynom operator*(int const&, polynom const&);

polynom operator-(polynom const&, polynom const&);
polynom operator-(polynom const&, double const&);
polynom operator-(double const&, polynom const&);
polynom operator-(polynom const&, int const&);
polynom operator-(int const&, polynom const&);

polynom operator/(polynom const&, double const&);
polynom operator/(polynom const&, int const&);





class legendre : public polynom
{
protected:
  polynom monome;
  bool checkB;
  double uperBound;
  double lowerBound;

public:
  legendre();
  legendre(int,bool checkB=true, double m_lowerBound=-1,double m_uperBound=1);
  ~legendre();

  void defineOrder(int,double m_lowerBound=-1,double m_uperBound=1);
  polynom recursive(polynom,polynom,int);
  bool checkBound(double);
  double toUnit(double);
  double eval(double);
  double operator()(double);
};



#endif
