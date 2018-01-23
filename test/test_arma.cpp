#include <iostream>
#define DONT_USE_WRAPPER
#include <armadillo>
#include "poly_element.hpp"
#include "kernel_element.hpp"
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
        polyElement legen(order);
        kernelElement kern;

        FunctionTrain<kernelElement> testkernel(ranks,ninput,kern);
        FunctionTrain<polyElement> testpoly(ranks,ninput,legen);

        arma::vec input=arma::randu<arma::vec>(ninput);

        double testValue=testkernel.eval(input);
        std::cout << testValue << std::endl;

        arma::vec testJac=testkernel.jacobian(input);
        testJac.print("Jacobian");

        arma::vec testGradwrtParam=testkernel.returnGradwrtParameters(input);
        testGradwrtParam.print("gradwrtParam");


        
        return 0;
}
