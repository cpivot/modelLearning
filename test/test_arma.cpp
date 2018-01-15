#include <iostream>
#define DONT_USE_WRAPPER
#include <armadillo>
#include "FT.hpp"

using namespace std;
using namespace arma;

int main()
{
        arma::vec ranks=arma::ones(3);
        ranks.fill(3);

        int ninput=3;
        int order=3;

        FunctionTrain test(ranks,ninput,order);

        arma::vec input=arma::randu<arma::vec>(3);
        double testValue=test.eval(input);

        std::cout << testValue << std::endl;
        return 0;
}
