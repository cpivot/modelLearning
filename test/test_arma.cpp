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
        polyElement legen;
        kernelElement kern;

        FunctionTrain<kernelElement> testkernel(ranks,ninput,kern);
        FunctionTrain<polyElement> testpoly(ranks,ninput,legen);

        testkernel.defineElement(order);
        testkernel.evaluateNumberOfParameters();
        testkernel.initialize(0.5);


        testpoly.defineElement(order);
        testpoly.evaluateNumberOfParameters();
        testpoly.initialize(0.5);



        arma::vec input=arma::randu<arma::vec>(ninput);
        input.fill(0.5);
        //testpoly.randomize();

        // cout << "******* Kernel ********" << endl;
        std::cout << testkernel.eval(input) << std::endl;
        std::cout << std::endl << std::endl << std::endl;
        testkernel.jacobian(input).print("Jacobian");
        std::cout << std::endl << std::endl << std::endl;
        testkernel.returnGradwrtParameters(input).print("gradwrtParam");

        // cout << "******* Legendre polynomials ********" << endl;
        // std::cout << testpoly.eval(input) << std::endl;
        // std::cout << std::endl << std::endl << std::endl;
        // testpoly.jacobian(input).print("Jacobian");
        // std::cout << std::endl << std::endl << std::endl;
        // testpoly.returnGradwrtParameters(input).print("gradwrtParam");

        return 0;
}
