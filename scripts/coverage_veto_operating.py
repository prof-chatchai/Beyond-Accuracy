"""False-Fail rate of the revised rule's coverage veto for an exactly valid split-conformal procedure.

Training-conditional coverage of split conformal is Beta(l, n_cal+1-l), l = ceil((n_cal+1)(1-alpha))
(Vovk 2013); test coverage is Binomial(n_test, coverage). The veto fires when the one-sided 95%
Clopper-Pearson lower bound is below 1 - alpha - eps.
Output: results/coverage_veto_operating.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import beta

ALPHA, EPS, R = 0.10, 0.02, 200_000
rng = np.random.default_rng(0)
rows = []
for n in (126, 317, 1800):
    l = int(np.ceil((n + 1) * (1 - ALPHA)))
    cov = rng.beta(l, n + 1 - l, R)
    k = rng.binomial(n, cov)
    lcb = np.where(k > 0, beta.ppf(0.05, k, n - k + 1), 0.0)
    rows.append({"n_cal": n, "n_test": n, "alpha": ALPHA, "eps": EPS,
                 "false_fail_rate": float((lcb < 1 - ALPHA - EPS).mean()), "draws": R})
pd.DataFrame(rows).to_csv("results/coverage_veto_operating.csv", index=False)
print(pd.DataFrame(rows))
