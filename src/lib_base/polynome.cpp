#include "polynome.hpp"


polynom::polynom()
{
  order=0;
  coef=ones(order+1);
}

polynom::~polynom()
{

}

void polynom::define(vec inputCoef)
{
  order=inputCoef.n_elem-1;
  coef.resize(inputCoef.n_elem);
  coef=inputCoef;
}

void polynom::define(int inputOrder)
{
  order=inputOrder;
  coef.resize(inputOrder+1);
  coef.fill(0.0);
}

void polynom::reset()
{
  coef.fill(0.0);
}

void polynom::affiche()
{
  //coef.print("coefficient");
  //cout << order << endl;
  for (int ii=0;ii<order+1;ii++)
    {
      if (coef(ii)!=0)
        cout << coef(ii) << "x^" << ii;
      if (ii<order)
        if (coef(ii+1)>0)
          cout << "+";

    }
  cout << endl;
  //cout << "toto" << endl;
}

double polynom::eval(double x)
{
  double result=coef(0);
  for (int ii=1;ii<order+1;ii++)
    result+=coef(ii)*std::pow(x,ii);
  return result;
}

double polynom::operator()(double x)
{
  return this->eval(x);
}

vec polynom::returnCoef()
{
  return coef;
}

void polynom::updateCoef(int whatOrder,double whatValue)
{
  if (whatOrder<=order)
    coef(whatOrder)=whatValue;
  else
    cout << "Order to high" << endl;
}

void polynom::updateCoef(vec updateCoef)
{
  if (order==updateCoef.n_elem)
    coef=updateCoef;
  else
    cout << "Wrong update vector" << endl;
}

int polynom::returnOrder()
{
  return order;
}

void polynom::updateOrder(int newOrder)
{
  int oldOrder=order;
  order=newOrder;
  coef.resize(order+1);
  for (int ii=oldOrder;ii<order-1;ii++)
    coef(ii)=0;
}

polynom polynom::derive()
{
  polynom derivePoly;

  if (order==0)
    {
      derivePoly.updateOrder(0);
      derivePoly.updateCoef(0,0.);
      return derivePoly;
    }

  derivePoly.define(order-1);
  //derivePoly.updateOrder(order-1);

  for (int ii=0;ii<order;ii++)
    derivePoly.updateCoef(ii,(ii+1)*coef(ii+1));

  return derivePoly;
}



polynom polynom::integrate()
{
  polynom integratePoly;

  integratePoly.define(order+1);
  integratePoly.updateOrder(order+1);
  for (int ii=1;ii<=order+1;ii++)
    integratePoly.updateCoef(ii,coef(ii-1)/(ii+1));

  return integratePoly;
}


void polynom::d()
{
  polynom derivePoly=derive();
  define(derivePoly.returnCoef());
}

/*
  void polynom::integrate()
  {
  polynom integratePoly=integrate();

  updateOrder(integratePoly.returnOrder());
  updateCoef(integratePoly.returnCoef());
  ~integratePoly();
  }
*/







void polynom::operator=(polynom const& first)
{
  define(first.coef);
}


void polynom::add(polynom second)
{
  int orderSecond=second.returnOrder();
  vec coefSecond=second.returnCoef();

  if (order>orderSecond)
    {
      for (int ii=0;ii<orderSecond;ii++)
        coef(ii)+=coefSecond(ii);
    }
  else
    {
      updateOrder(orderSecond);
      for(int ii=0;ii<order;ii++)
        coef(ii)+=coefSecond(ii);
    }
}

void polynom::add(double value)
{
  polynom monome;
  monome.define(1);
  monome.updateCoef(0,value);
  add(monome);
}


polynom operator+(polynom const& first,polynom const& second)
{
  polynom third=first;
  third.add(second);
  return third;
}


polynom operator+(polynom const& first, double const& value)
{
  polynom monome;
  monome.define(1);
  monome.updateCoef(0,value);
  return first+monome;
}

polynom operator+(double const& value, polynom const& first)
{
  return first+value;
}

polynom operator+(polynom const& first, int const& value)
{
  return (double)value + first;
}

polynom operator+( int const& value, polynom const& first)
{
  return (double)value + first;
}


void polynom::operator+=(polynom const& second)
{
  add(second);
}

void polynom::operator+=(double const& value)
{
  add(value);
}

void polynom::operator+=(int const& value)
{
  add((double)value);
}


void polynom::multiply(polynom second)
{
  vec prodCoef=zeros(order+second.returnOrder()+1);
  vec coefSecond=second.returnCoef();

  for (int ii=0;ii<order+1;ii++)
    for (int jj=0;jj<second.returnOrder()+1;jj++)
      prodCoef(ii+jj)+=coef(ii)*coefSecond(jj);

  define(prodCoef);
}

void polynom::multiply(double value)
{
  polynom monome;
  monome.define(1);
  monome.updateCoef(0,value);
  multiply(monome);
}


polynom operator*(polynom const& first, polynom const& second)
{
  polynom third=first;
  third.multiply(second);
  return third;

}

polynom operator*(polynom const& first, double const& value)
{
  polynom monome;
  monome.define(1);
  monome.updateCoef(0,value);
  polynom firstCopy=first;
  firstCopy.multiply(monome);
  return firstCopy;
}

polynom operator*(double const& value,polynom const& first)
{
  return first*value;
}

polynom operator*(polynom const& first, int const& value)
{
  return (double)value*first;
}

polynom operator*(int const& value, polynom const& first)
{
  return (double)value*first;
}


void polynom::operator*=(polynom const& second)
{
  multiply(second);
}

void polynom::operator*=(double const& value)
{
  multiply(value);
}

void polynom::operator*=(int const& value)
{
  multiply((double)value);
}

polynom operator-(polynom const& first,polynom const& second)
{
  return first+second*(-1.);
}

polynom operator-(polynom const& first,double const& value)
{
  return first+value*(-1);
}

polynom operator-(double const& value, polynom const& first)
{
  return value+(-1.)*first;
}

polynom operator-(polynom const& first, int const& value)
{
  return first+(-1.)*(double)value;
}

polynom operator-(int const& value, polynom const& first)
{
  return value+(-1.)*first;
}

polynom operator/(polynom const& first, double const& value)
{
  return first*(1./value);
}

polynom operator/(polynom const& first, int const& value)
{
  return first/((double)value);
}





legendre::legendre()
{

}

legendre::legendre(int m_order,bool checkB_m, double m_lowerBound,double m_uperBound)
{
  defineOrder(m_order, m_lowerBound,m_uperBound);
  //coef.resize(m_order);
  coef=coef.head(m_order+1);
  checkB=checkB_m;
}

void legendre::defineOrder(int m_order, double m_lowerBound,double m_uperBound)
{

  lowerBound=m_lowerBound;
  uperBound=m_uperBound;

  polynom Pa,Pb,Pc;
  monome.define(1);
  monome.updateCoef(0,0);
  monome.updateCoef(1,1);
  //  monome.affiche();

  Pa.define(0);
  Pa.updateCoef(0,1);

  Pb=monome;

  if (m_order==0)
    define(Pa.returnCoef());
  if (m_order==1)
    define(Pb.returnCoef());

  if (m_order>1)
    for (int ii=1;ii<m_order;ii++)
      {
        Pc=recursive(Pb,Pa,ii);
        arma::vec coef=Pc.returnCoef();
        define(coef.head(ii+2));
        Pa=Pb;
        Pb=Pc;
      }
}

legendre::~legendre()
{

}

polynom legendre::recursive(polynom Pn, polynom Pnb,int currentOrder)
{
  return ((2*currentOrder+1)*monome*Pn-currentOrder*Pnb)/(currentOrder+1);
}

bool legendre::checkBound(double value)
{
  //  std::cout << lowerBound << " " << value << " " << uperBound << std::endl;
  if (value<uperBound && value>lowerBound)
    return true;
  else
    return false;
}

double legendre::toUnit(double value)
{
  return 0.5*value/(uperBound-lowerBound)-(uperBound+lowerBound)/(uperBound-lowerBound);
}


double legendre::eval(double value)
{
  if (checkB)
  {
    if(checkBound(value))
      return polynom::eval(toUnit(value));
    else
    {
      cout << "Value out of bounds" << endl;
      return 1./0.0;
    }
  }
  else
    return polynom::eval(value);
}

double legendre::operator()(double x)
{
  return this->eval(x);
}
