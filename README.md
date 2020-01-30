# BNQD
A Python toolbox for Bayesian Nonparametric Quasi-Experimental Design. 

The de facto standard for causal inference is the randomized controlled trial, where one compares a manipulated group with a control group in order to determine the effect of an intervention. However, this research design is not always realistically possible due to pragmatic or ethical concerns. In these situations, quasi-experimental designs (QEDs may provide a solution, as these allow for causal conclusions at the cost of additional design assumptions. 

In this repository, we provide the implementation of a Bayesian non-parametric model-comparison-based take on QED, called BNQD. It quantifies (the presence of) an effect using a Bayes factor and and Bayesian model averaged posterior distribution. For basic usage, see the demo.py script, and for slightly more intricate examples, see the empirical examples and simulations.

## Literature

* Max Hinne, Marcel van Gerven and Luca Ambrogioni, 2019. Causal inference using Bayesian non-parametric quasi-experimental design. ArXiv: https://arxiv.org/abs/1911.06722.
