# BNQD
A Python toolbox for Bayesian Nonparametric Quasi-Experimental Design. 

The de facto standard for causal inference is the randomized controlled trial, where one compares a manipulated group with a control group in order to determine the effect of an intervention. However, this research design is not always realistically possible due to pragmatic or ethical concerns. In these situations, quasi-experimental designs (QEDs) may provide a solution, as these allow for causal conclusions at the cost of additional design assumptions. 

In this repository, we provide the implementation of a Bayesian non-parametric model-comparison-based take on QED, called BNQD. It quantifies (the presence of) an effect using a Bayes factor and and Bayesian model averaged posterior distribution. For basic usage, see the BNQD demo.ipynb notebook.

The current implementation of BNQD was tested using the following dependencies:

* [GPFlow](https://gpflow.readthedocs.io/en/master/index.html) 2.2.1
* TensorFlow 2.5.0
* NumPy 1.20.2

## Literature

* Max Hinne, David Leeftink, Marcel van Gerven and Luca Ambrogioni, 2022. Bayesian model averaging for nonparametric discontinuity
design. PLOS ONE 17(6): e0270310. https://doi.org/10.1371/journal.pone.0270310
* David Leeftink and Max Hinne, 2020. Spectral discontinuity design: Interrupted time series with spectral mixture kernels. Proceedings of the Machine Learning for Health NeurIPS Workshop, in PMLR 136:213-225. Available from http://proceedings.mlr.press/v136/leeftink20a.html.


