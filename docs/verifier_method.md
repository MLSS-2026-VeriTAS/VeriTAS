# The VeriTAS Verifier: Bayesian Validated Information Gain

This document gives the precise probabilistic model behind the VeriTAS Verifier.
It resolves the open definitions left as TODOs in the project README: what
"additional information contributed by the most recent iteration" means, the
prior/posterior quantities involved, the decision rule, and how the additional
score is derived.

## 1. Problem and motivation

An iterative ML-research agent (such as MLRC-Bench) proposes, at each iteration
`t`, a modification to its current best method and obtains one or more
*objective* measurements of performance. These measurements are **noisy**: seeds,
stochastic optimization, and data splits all induce run-to-run variance. A naive
controller that adopts any iteration whose observed score beats the incumbent is
routinely fooled by lucky runs, and then *compounds* the mistake by building
subsequent work on a method that was never actually better.

The Verifier's job is to assign each iteration an **additional score** that
reflects how much *genuine, reproducible* progress it contributed, and a
**decision** about whether to credit (accept) the iteration. Both must be robust
to measurement noise.

## 2. Observation model

Let `theta_t` denote the latent, noise-free performance of the candidate method
at iteration `t`. We observe `k_t >= 1` measurements

```
y_{t,i} = theta_t + eps_{t,i},   eps_{t,i} ~ N(0, sigma^2),   i = 1..k_t
```

with unknown measurement variance `sigma^2`. (Scores are oriented so that larger
is better; metrics where lower is better are negated.)

## 3. Conjugate Normal-Inverse-Gamma posterior

We place a conjugate Normal-Inverse-Gamma (NIG) prior on `(theta, sigma^2)`:

```
sigma^2        ~ Inverse-Gamma(alpha_0, beta_0)
theta | sigma^2 ~ N(mu_0, sigma^2 / lambda_0)
```

Given measurements with sample mean `ybar` and sum of squared deviations
`S = sum_i (y_i - ybar)^2`, the posterior is NIG with

```
lambda_n = lambda_0 + k
mu_n      = (lambda_0 * mu_0 + k * ybar) / lambda_n
alpha_n   = alpha_0 + k/2
beta_n    = beta_0 + S/2 + (lambda_0 * k / lambda_n) * (ybar - mu_0)^2 / 2
```

The **marginal posterior** over the mean is Student-t:

```
theta | data ~ t_{2 alpha_n}( mu_n,  beta_n / (lambda_n * alpha_n) )
```

with location `mu_n`, scale `sqrt(beta_n / (lambda_n alpha_n))`, and degrees of
freedom `2 alpha_n`. Keeping `alpha_0 >= 1` guarantees `dof > 2`, so the marginal
has a finite variance

```
Var[theta | data] = scale^2 * dof / (dof - 2).
```

The posterior mean of the **measurement noise** is `E[sigma^2] = beta_n /
(alpha_n - 1)` for `alpha_n > 1`; we write its square root as `sigma_hat`.

Implementation: `nig_update`, `student_t_marginal`, `gaussian_moments`,
`expected_noise_std` in [`veritas/verifier/model.py`](../veritas/verifier/model.py).

## 4. The incumbent and the improvement

The Verifier maintains a Gaussian belief `N(m*, v*)` over the best achievable
("incumbent") performance `theta*`. For the candidate at iteration `t`, summarize
its Student-t marginal by Gaussian moments `(m_c, v_c)`. Treating candidate and
incumbent as independent, the **improvement**

```
Delta_t = theta_t - theta*_{t-1}   ~  N(mu_D, sigma_D^2),
mu_D = m_c - m*,   sigma_D^2 = v_c + v*.
```

## 5. The three quantities per iteration

### 5.1 Probability of improvement (decision rule)

We require an improvement to exceed a minimal effect size `eps_t` that scales
with the *inferred measurement noise*, so a noisy candidate must show a larger
mean gain to be credited:

```
eps_t = max(eps_min, eps_min_rel * sigma_hat)
POI_t = P(Delta_t > eps_t) = Phi( (mu_D - eps_t) / sigma_D ).
```

The iteration is **accepted** iff `POI_t > tau`. This is a Bayesian one-sided
test that directly controls credited false improvements.

### 5.2 Validated information gain (the additional score)

We define the information an iteration contributes as one-sided evidence that it
genuinely improved on the incumbent, i.e. a KL-style score against the
no-improvement null:

```
IG_t = KL( N(max(mu_D, 0), sigma_D^2) || N(0, sigma_D^2) )
     = max(mu_D, 0)^2 / (2 sigma_D^2).
```

This is a one-sided squared signal-to-noise ratio of the measured improvement:
it is non-negative, exactly zero for regressions, near zero for within-noise or
highly uncertain iterations, and large only for clear gains measured with low
uncertainty. It is well conditioned (unlike a KL between beliefs of very
different variance, which can diverge when uncertainty inflates).

### 5.3 Expected improvement (diagnostic)

For interpretability we also report the Bayesian-optimization Expected
Improvement acquisition value with the incumbent as reference:

```
EI_t = E[max(Delta_t, 0)] = mu_D * Phi(z) + sigma_D * phi(z),   z = mu_D / sigma_D.
```

## 6. The reward and the incumbent update

The additional VeriTAS score for the iteration is

```
r_t = POI_t * IG_t,
```

the validated information gain weighted by the confidence that the gain is real.
Unconfident noise spikes (low `POI`) and within-noise moves (low `IG`) both
receive `~0`; confident, reproducible gains are rewarded in proportion to their
signal-to-noise ratio.

The deployed frontier advances only on acceptance: when `POI_t > tau`, the
accepted candidate's posterior `(m_c, v_c)` becomes the new incumbent belief;
otherwise the incumbent is left unchanged, so a single lucky run cannot corrupt
the running belief about the best method.

Implementation: [`veritas/verifier/verifier.py`](../veritas/verifier/verifier.py).

## 7. Calibrated fusion of the LLM-as-a-Judge rubric

MLRC-Bench's `LLM_as_a_Judge` rates a method on a 1..5 Likert scale across
dimensions (Clarity, Validity, Rigorousness, Innovativeness, Generalizability).
Rather than discard these subjective signals or trust them blindly, the Verifier
turns them into a **prior** over the candidate's performance which the objective
measurements then confirm or override.

Given the mean rubric rating `rbar` over the available dimensions, normalize it
to `[-1, 1]` and map it to an expected gain via a calibrated affine map:

```
normalized = (rbar - 3) / 2
expected_gain = slope * normalized + intercept
mu_0 = m* + expected_gain
```

The prior strength (pseudo-count `lambda_0`) grows with rubric *agreement*:
`lambda_0 = rubric_prior_strength / (1 + disagreement)`, where `disagreement` is
the standard deviation of the per-dimension ratings. The `slope`/`intercept` are
fit by least squares from historical `(mean rating, observed gain)` pairs
(`calibrate_rubric_map`), so the prior is grounded in data rather than guessed.
Calibration of `POI` itself is assessed with reliability diagrams
(`reliability_table`).

Implementation: [`veritas/verifier/fusion.py`](../veritas/verifier/fusion.py).

## 8. Validation protocol and net improvement

The Verifier is validated by a paired ablation that holds all randomness fixed
(common random numbers) and varies only the acceptance policy:

* **greedy**   - adopt a candidate whenever its mean observed score beats the
                 current method (the standard baseline);
* **verifier** - adopt a candidate only when `POI_t > tau`.

Because the synthetic environment exposes each candidate's *true* effect, we can
measure quantities a real run cannot:

```
net improvement = (verifier final true performance) - (greedy final true performance)
mean regret     = average over iterations of (oracle - deployed true performance)
false-accept rate = fraction of accepted iterations whose true effect <= 0
```

reported per seed and in aggregate (mean +/- standard deviation). A positive net
improvement and lower regret/false-acceptance indicate the Verifier improves the
agent under a controlled configuration. The same metrics carry over to real
MLRC-Bench runs by replacing the synthetic generator with the log adapter in
[`veritas/verifier/mlrc_adapter.py`](../veritas/verifier/mlrc_adapter.py); see
[`scripts/validate_verifier.py`](../scripts/validate_verifier.py).

## 9. Symbol reference

| Symbol | Meaning |
|---|---|
| `theta_t` | latent (noise-free) performance of the candidate at iteration `t` |
| `theta*` | latent performance of the incumbent (best) method |
| `sigma^2`, `sigma_hat` | measurement-noise variance and its posterior-mean std |
| `Delta_t` | improvement `theta_t - theta*_{t-1}` |
| `eps_t` | minimal effect size that counts as an improvement |
| `tau` | acceptance threshold on `POI_t` |
| `POI_t` | probability of (genuine) improvement |
| `IG_t` | validated information gain (nats) |
| `EI_t` | expected improvement |
| `r_t` | additional VeriTAS score, `POI_t * IG_t` |
