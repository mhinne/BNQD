# To implement

We wish to extend BNQD with several additional features. In particular, we wish to exploit the expressiveness of GPs 
compared to (locally) linear models that are common in RD, and moving averages in ITS.

* 1D RD/ITS with covariates.
* Multi-response RD.
* Multi-response ITS.
* Multi-input RD.
* Multi-input ITS.
* Add covariates to either.
* A generic approach for the Laplace approximation of the marginal likelihood.

## Regression discontinuity design with covariates

- See also Frölich & Huber, 2019.
- Further important reading: https://towardsdatascience.com/get-a-grip-when-to-add-covariates-in-a-linear-regression-f6a5a47930e5

# To think about

* What is a good effect size measure for ITS? 
  Do we extrapolate the pre-threshold continuous vs pre-threshold discontinuous model? 
  Does the same formalism still make sense?
* Applications.
* Add graph kernels for discontinuous connectivity?
* Per reviewer 3 at UAI: in ITS, we have instead the effect of the treatment on the treated, not an average causal 
  effect (since we have no control time series that is not subject to treatment)



---
# Things to do / not to forget:

0. REMOVE SPLIT MODELS AND JUST USE INDEPENDENT KERNELS WITH/WITHOUT SAME BASE KERNEL - check
1. Check BIC penalty in ITS (M1 has more parameters). - check
2. Multidimensional input and output.
3. Sinc and Minecraft kernels; the latter solves an issue with the spectral mixture.
4. Additional focus on extrapolation for ITS; also add BMA predictions. Implement as extrapolate_y(x_new)?
5. Laplace approximation for evidence, to replace BIC.
6. Check IndependentKernel for BNQD with mode='ITS'. - check
7. For ITS, focus on regime between x0 and the reverting to the mean.
8. Maybe first to MOGP-RD; as a solution for incorporating covariates?
9. SwitchedMeanFunction for different means per X
10. Sequential ITS with WISKI?

# Issues

For M0, prediction automatically takes into account the existing observations, which are also >x0

# Study ideas:

- Focus on incremental (MOGP) ITS?
---

