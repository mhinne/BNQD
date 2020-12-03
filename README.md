# BNQD
A Python toolbox for Bayesian Nonparametric Quasi-Experimental Design. 

The de facto standard for causal inference is the randomized controlled trial, where one compares a manipulated group with a control group in order to determine the effect of an intervention. However, this research design is not always realistically possible due to pragmatic or ethical concerns. In these situations, quasi-experimental designs (QEDs) may provide a solution, as these allow for causal conclusions at the cost of additional design assumptions. 

In this repository, we provide the implementation of a Bayesian non-parametric model-comparison-based take on QED, called BNQD. It quantifies (the presence of) an effect using a Bayes factor and and Bayesian model averaged posterior distribution. For basic usage, see the BNQD demo.ipynb notebook.

The current implementation of BNQD was tested using the following dependencies:

* [GPflow](https://gpflow.readthedocs.io/en/master/index.html) 2.0.0
* tensorflow 2.1.0
* tensorboard 2.1.1
* tensorflow_probability 0.9
* ptable 0.9.2

## Literature

* Max Hinne, Marcel van Gerven and Luca Ambrogioni, 2020. Causal inference using Bayesian non-parametric quasi-experimental design. ArXiv: https://arxiv.org/abs/1911.06722.
