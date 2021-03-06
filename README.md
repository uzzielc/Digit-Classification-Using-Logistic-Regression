# Digit-Classification-Using-Logistic-Regression

## Description

The data set used is a subset of the MNIST dataset and contains the digits 4 & 9.
The images are stored as column vectors and labels are included.
For classification, I have implemented Newton’s method (a.k.a. Newton-Raphson) to find a minimizer of J(θ) = −l(θ) + λ||θ||^2
i.e. L2-Regularized Negative Log-Likelihood.

## Performance
The data is split into 2000 training samples and 1000 testing samples. 
The algorithm misclassifies 48 out of the 1000 testing samples i.e. achieves a correct classification rate of ~94%
