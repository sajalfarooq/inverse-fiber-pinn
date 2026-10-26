# Inverse fiber modelling technical report

## Research question and findings

Can phase-resolved observations of a known input pulse identify two NLSE coefficients, and does adding a physics residual improve prediction compared with a data-only network? The retained experiment compares classical numerical fitting, a joint inverse PINN, data-only learning, and a known-coefficient physics-only forward model. It uses independent network seeds and validates at four distances absent from supervised training.

The conventional numerical fit remains the accuracy baseline. The data-only network has a slightly lower mean pulse-region field error than the joint PINN on this configuration. This does not imply that physics constraints are generally harmful: the loss weights, collocation points, finite training budget and matched measurement model all affect the result. Reporting the ablation prevents assigning every improvement to the PDE term.

## Forward and inverse models

The normalized equation is $iu_\xi+(d/2)u_{\tau\tau}+g|u|^2u=0$. Relative to a fixed reference length and time, d is a positive dispersion magnitude for the anomalous regime and g is the nonlinear coefficient. True values are (0.9,1.2). The input $u_0=\exp(-\tau^2/2+0.25i\tau^2)$ is known exactly.

Measurements sample 32 temporal positions at xi=0.25,0.5,0.75,1, producing 128 complex labels. Added circular complex noise has absolute standard deviation 0.01, split equally in variance between real and imaginary components. This assumes coherent field information; intensity-only measurements would define a different inverse problem with different ambiguities.

Data generation uses 1000 steps per distance unit. Numerical fitting uses 200 steps per unit, reducing a discretization inverse crime while retaining the same physical equation and temporal grid. The baseline noise study uses several amplitudes and repeats; it is distinct from the fixed shared dataset used for advanced ablations.

Classical fitting minimizes the concatenation of real and imaginary residuals with positive bounded coefficients. A local Jacobian measures sensitivity near the fitted solution. It does not establish unique recovery over all coefficient pairs or other pulse families.

## PINN derivation and training

Writing u=a+ib gives two residuals:

$$r_R=-b_\xi+\frac d2a_{\tau\tau}+g(a^2+b^2)a,$$
$$r_I=a_\xi+\frac d2b_{\tau\tau}+g(a^2+b^2)b.$$

Automatic differentiation computes derivatives of the two-output tanh network with respect to original coordinates. Scaling occurs inside the network, so chain-rule factors are retained. Three hidden layers of width 48 are used. Learned log coefficients enforce positivity. An analytical soliton and nonlinear plane wave check the residual signs independently of optimization.

Joint training minimizes physics MSE + 20 times label MSE + 20 times initial-condition MSE + boundary MSE. The standard budget is 3000 Adam steps followed by 40 L-BFGS outer steps. Adam resamples collocation points, while L-BFGS uses fixed points so its closure has a deterministic objective. Full per-term losses and coefficient trajectories are retained.

The data-only ablation uses the same architecture, labels, known initial condition and boundary prior, but no PDE loss. Its coefficients are frozen and are not estimators. The physics-only ablation uses the true coefficients; without data an unknown-coefficient inverse problem is not identified by the initial condition alone. Only one physics-only seed is run, so its result is illustrative.

## Held-out evaluation

Validation distances are 0.125,0.375,0.625,0.875, all within the training domain but absent from the labels. This tests interpolation in propagation distance, not extrapolation or transfer to unseen pulses. Pulse-region error is relative complex-field L2 over abs(tau)<4. A full-window error is also saved so errors in low-intensity regions remain visible.

| Model | Seeds | Mean pulse field error | Sample standard deviation |
|---|---:|---:|---:|
| joint_inverse | 3 | 1.245% | 0.220% |
| data_only | 3 | 1.135% | 0.044% |
| physics_only_known_coefficients | 1 | 3.251% | 0.000% |

The classical fit's unseen-distance pulse error is **0.235%**, with fitted coefficients 0.891249, 1.199386.

| Joint PINN seed | Recovered d | Recovered g |
|---|---:|---:|
| 7 | 0.867586 | 1.184023 |
| 19 | 0.870420 | 1.185119 |
| 31 | 0.870620 | 1.184559 |

## Computational cost and reproducibility

Training/parameter-fitting time is stored separately from forward evaluation time. Five repeated forward timings evaluate the same four distance planes and 128 temporal samples. The SSFM uses 400 steps per distance unit; the PINN includes tensor conversion. These models have different accuracy, so timing differences are not an equal-error speedup claim. Timing depends on hardware, library versions, CPU load and warm-up.

Run `python reproduce.py --profile standard`. Saved checkpoints include architecture width, seed, optimizer budget, profile and mode. `python verify.py` reloads every retained checkpoint from the frozen release and recalculates its validation error without retraining. Three notebooks cover the baseline, residual/checkpoint and ablation interpretation.

## Limits and next work

Only one shared noisy measurement dataset and one pulse family are used for the seed comparison. Training-seed variability is not measurement uncertainty. The finite-window zero-boundary prior differs from exact periodic SSFM boundaries but is suitable only while the pulse is localized. Broader measurements, intensity-only labels, mismatched input amplitude, different coefficient regimes and more seeds remain future investigations. The work demonstrates a transparent inverse-study pipeline; it does not establish a generally superior neural solver.

## References

1. G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed., Academic Press (2019), Chapters 2–5. [Publisher](https://shop.elsevier.com/books/nonlinear-fiber-optics/agrawal/978-0-12-817042-7). Physical model, normalization and nonlinear pulse propagation.
2. M. Raissi, P. Perdikaris and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics* **378**, 686–707 (2019). [DOI](https://doi.org/10.1016/j.jcp.2018.10.045). PINN methodology and forward/inverse formulations.
3. M. Raissi, P. Perdikaris and G. E. Karniadakis, *Physics Informed Deep Learning Part II: Data-driven Discovery of Nonlinear Partial Differential Equations* (2017). [Paper](https://arxiv.org/abs/1711.10566). Background on learning unknown PDE coefficients.
4. A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby and M. W. Mahoney, “Characterizing possible failure modes in physics-informed neural networks,” *Advances in Neural Information Processing Systems* **34** (2021). [Paper](https://arxiv.org/abs/2109.01050). Motivation for inspecting optimization failures and ablations; this project does not reproduce that paper's experiments.
5. SciPy developers, [least_squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html). Implementation reference for the bounded nonlinear least-squares baseline.
