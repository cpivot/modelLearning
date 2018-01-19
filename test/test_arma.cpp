#include <iostream>
#define DONT_USE_WRAPPER
#include <armadillo>
#include "adam.hpp"
#include "FT.hpp"

using namespace std;
using namespace arma;

int main()
{
        int ninput=3;

        arma::vec ranks;
        //ranks << 2 << 4 << 3 << 2;
        ranks << 3 << 3;

        int order=2;

        FunctionTrain test(ranks,ninput,order,0.5);

        arma::vec input=arma::randu<arma::vec>(ninput);

        double testValue=test.eval(input);
        std::cout << testValue << std::endl;

        arma::vec testJac=test.jacobian(input);
        testJac.print("Jacobian");

        arma::vec testGradwrtParam=test.returnGradwrtParameters(input);
        testGradwrtParam.print("gradwrtParam");
        return 0;
}
