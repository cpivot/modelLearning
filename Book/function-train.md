# Function trains

## Definition


$$
f(\mathbf{x})=\mathcal{F}_1(x_1) \mathcal{F}_2(x_2) \cdots \mathcal{F}_n(x_n)
$$


where


$$
\mathcal{F}_k(x_k) : \mathbb{R} \to \mathbb{R}^{r_k,r_{k+1}}
$$



$$
\mathcal{F}_k(x_k)=\begin{pmatrix} f_{1,1}^k(x_k) & \cdots & f_{1,r_{k+1}}^k(x_k) \\ \vdots & & \vdots \\ f_{r_{k},1}^k(x_k) & \cdots &  f_{r_{k},r_{k+1}}^k(x_k)\end{pmatrix}
$$


The rank vector can be written as


$$
\mathbf{r}=\lbrace 1,r_1,r_2,\cdots,r_n,1 \rbrace \in \mathbb{R}^{n+2}
$$


In the code, only the $$n$$ value of the center are necessary, you don't need do add the first and the last ones

```cpp
int ninput=4;
arma::vec ranks=arma::zeros(ninput);
ranks << 2 << 5 << 3;
```

Function $$f_{i,j}^k(x_k)$$ can be choosen between polynomials and kernels :

* Kernels $$f_{i,j}(x_k)= \theta_1 \exp{- \dfrac{\left(x_k-\theta_2\right)^2}{\theta_3}}$$

```cpp
kernelElement kern;
FunctionTrain<kernelElement> FTKernel(ranks,ninput,kern);
```

* Polynomials, $$f^k_{i,j}(x_k) = \sum_{p=1}^d \theta_p \phi_p{x_k}$$ currently only the Legendre polynomials are implemented 

```cpp
PolyElement legen;
FunctionTrain<kernelElement> FTpoly(ranks,ninput,legen);
```

Finally, we need to define the elements, then evaluate the number of parameters and intialize with a value :

```cpp
FTkernel.defineElement(order);
FTkernel.evaluateNumberOfParameters();
FTkernel.initialize(0.5);
```

Since the code uses templates, we_ need to send a order even for kernels \(not use\)._

## Evaluate, Jacobian, etc

The code allows to evaluate :

* The value at a given point $$\widetilde{f}(\mathbf{x})$$
* The jacobian at a given point $$\mathbf{\nabla}_{\mathbf{X}}\widetilde{f}(\mathbf{x})$$
* The gradient with respect to parameters at a given point $$\mathbf{\nabla}_{\mathbf{\theta}}\widetilde{f}(\mathbf{x})$$

```cpp
arma::vec input=arma::randu<arma::vec>(ninput);

cout << FTkernel(input) << endl;
cout << endl << endl << endl;

FTkernel.jacobian(input).print("Jacobian");
cout << endl << endl << endl;

FTkernel.returnGradwrtParameters(input).print("gradwrtParam");
```

## Update parameters

```cpp
int numberOfParameters=KTkernel.returnNumberOfParameters();
arma::vec updateVector=arma::randn(numberOfParameters);
KTkernel.updateParameters(updateVector);
```

This code juste perform 

$$\theta \leftarrow \theta+ \Delta \theta$$

