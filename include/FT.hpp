#include <iostream>
#include <armadillo>
#include <math.h>


#ifndef ____FT__
#define ____FT__

class FunctionTrain
{
private:
arma::vec ranks;
int order;
int ninput;

public:
FunctionTrain(arma::vec, int,int);     //ranks,input,order
double eval(arma::vec);
};

#endif
