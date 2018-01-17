#include <iostream>
#define DONT_USE_WRAPPER
#include <armadillo>
#include "FT.hpp"

using namespace std;
using namespace arma;

int main()
{
        int ninput=5;

        arma::vec ranks;
        ranks << 2 << 4 << 3 << 2;

        int order=2;

        FunctionTrain test(ranks,ninput,order,0.5);

        arma::vec input=arma::randu<arma::vec>(ninput);
        double testValue=test.eval(input);

        std::cout << testValue << std::endl;
        return 0;
}
