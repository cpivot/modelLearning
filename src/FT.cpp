#include "FT.hpp"

#define PI 3.14159265


FunctionTrain::FunctionTrain(arma::vec ranks_m,int ninput_m, int order_m  )
{
        ninput=ninput_m;
        order=order_m;
        ranks=arma::ones(ninput+2);
        ranks.subvec(1,ninput)=ranks_m;
}

double FunctionTrain::eval(arma::vec input)
{
        double value=0;

        return value;
}
