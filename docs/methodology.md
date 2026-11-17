# Methodology

This page summarises the pipeline that the code implements. The paper gives the full derivations, propositions and proofs.

## 1. Corruption

Each test engine contributes its last window $X \in \mathbb{R}^{30 \times 14}$. A binary mask $M$ marks injected gaps. Gap starts are drawn per cycle and per channel, and each start removes a burst of three consecutive cycles (`corruption.py`).

| Mechanism | Start probability for channel $j$ at cycle $t$ |
|---|---|
| MCAR | $r / 3$ |
| MAR | $2r / 3$ if the neighbouring channel $j+1$ is above its window median, else 0 |
| MNAR | $2r / 3$ if channel $j$ itself is above its window median, else 0 |

Rates are $r \in \{0.1, 0.3, 0.5\}$. The generator seed is a fixed function of mechanism, rate and seed, so masks are identical across runs.

## 2. Repair

Nine methods fill the gaps (`imputers.py`). MICE and CSDI return $m$ posterior draws ($m = 5$ and $m = 20$). The point estimate is the draw mean for MICE and the draw median for CSDI. *No repair* fills zeros, and *clean input* feeds the uncorrupted window as a reference.

## 3. Twin and ensemble

Each member $k$ predicts a mean $\mu_k$ and a variance $\sigma_k^2$ and is trained with the Gaussian negative log-likelihood after an MSE warm start (`twin.py`). The $K = 5$ members form a uniform mixture:

$$
\mu^\ast = \frac{1}{K}\sum_k \mu_k, \qquad
W = \frac{1}{K}\sum_k \sigma_k^2 + \operatorname{Var}_k(\mu_k).
$$

## 4. Rubin's rules

For the multiple imputers, the ensemble runs on every draw $i = 1, \dots, m$ and the results are pooled (`uncertainty.rubin_pool`):

$$
\bar\mu = \frac{1}{m}\sum_i \mu^\ast_i, \qquad
\bar W = \frac{1}{m}\sum_i W_i, \qquad
B = \frac{1}{m-1}\sum_i (\mu^\ast_i - \bar\mu)^2, \qquad
T = \bar W + \left(1 + \frac{1}{m}\right) B.
$$

The ablation replaces $T$ by $\bar W$ on the same draws (columns `*_noB` in the results).

## 5. Conformal calibration

On a clean random tenth of the training windows, the scores $s = |y - \mu^\ast| / \sqrt{W}$ give the multiplier $\hat q$ as the $\lceil (n+1)(1-\alpha) \rceil$-th smallest score with $\alpha = 0.05$ (`uncertainty.conformal_quantile`). The released runs gave $\hat q$ = 1.608, 1.853, 1.495 and 2.007 for FD001 to FD004. Intervals are $\mu \pm \hat q \sqrt{T}$.

## 6. Decisions and regret

An engine is critical when its true RUL is at most $\tau = 20$ cycles. With costs $C_{\mathrm{pm}} = 1$, $C_{\mathrm{fa}} = 2$ and $C_{\mathrm{cm}} = 10$ (`metrics.py`):

* the **point rule** alarms when $\mu \le \tau$;
* the **risk-aware rule** alarms when $\mu - \hat q \sqrt{T} \le \tau$;
* **regret** is the fleet cost of a rule minus the cost of perfect information, which equals $2\,\mathrm{FP} + 9\,\mathrm{FN}$.

Widening the margin lowers expected cost exactly when an extra alarm is critical with probability above $2/11$.

## 7. Statistics

`stats.py` holds every analysis in the paper:

* `two_layer_taus` and `divergence_test`: Kendall τ between imputation RMSE and RUL RMSE, PHM08 and regret in each of the 36 cells, with a one-sided Wilcoxon test on the gaps.
* `ranking_divergence`: per-cell τ with seed-bootstrap 95 % intervals.
* `policy_ratios`: point regret over risk-aware regret.
* `b_term_effect`: the B effect against coverage without B.
* `out_of_sample_selection`: select the repair method on seeds 0 to 4 and score it on seeds 5 to 9.
