

/*
Loss function are extract from Charbonnier et al
Edge-Preserving regularization
IEEE Transactions on image processing
vol 6 no 2 1997

The name comes from the original publication
*/



/*
S. Geman and D. E. McClure, “Bayesian image analysis: An application
to single photon emission tomography,” in Proc. Statistical Computation
Section, Amer. Statistical Assoc., Washington, DC, 1985, pp. 12–18.
*/
void phiGM(double error,double & value,double & derivate)
{
  double errorSquare=std::pow(error,2);
  value=errorSquare/(1+errorSquare);
  derivate=2*error/(std::pow(1+errorSquare,2));
}



/*
T. Hebert and R. Leahy, “A generalized EM algorithm for 3-D Bayesian
reconstruction from Poisson data using Gibbs priors,” IEEE Trans. Med.
Imag., vol. MI-8, pp. 194–202, June 1990.
*/

void phiHL(double error,double & value,double & derivate)
{
  double errorSquare=std::pow(error,2);
  value=std::log(1+errorSquare);
  derivate=2*error/(1+errorSquare);
}



/*
P. Charbonnier, G. Aubert, L. Blanc-F´eraud, and M. Barlaud, “Two deterministic
half-quadratic regularization algorithms for computed imaging,”
in Proc. 1st IEEE ICIP, Austin, TX, Nov. 1994.
*/

void phiHS(double error,double & value,double & derivate)
{
  double errorSquare=std::pow(error,2);
  value=2*std::sqrt(1+errorSquare)-2;
  derivate=2*error/(std::sqrt(1+errorSquare));
}


/*
P. J. Green, “Bayesian reconstructions from emission tomography data
using a modified EM algorithm,” IEEE Trans. Med. Imaging, vol. 9, pp.
84–93, Mar. 1990.
*/

void phiGR(double error,double & value,double & derivate)
{
  value=2*std::log(std::cosh(error));
  if (std::abs(error)<1e-6)
    derivate=2*error;
  else
    derivate=2*std::tanh(error);
}
