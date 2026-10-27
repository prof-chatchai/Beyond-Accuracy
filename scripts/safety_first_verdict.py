"""Re-score existing result files under the safety-first (non-compensatory) verdict rule.

No model is refitted: the rule is applied to per-seed metrics already written by
oracle_control.py, synthetic_robustness.py, model_sweep.py, operator_stress.py and
ktas_case_study.py. Outputs:
  results/safety_first_verdict_seed.csv     one row per (substrate, outcome, model, seed[, operator, level])
  results/safety_first_verdict_summary.csv  verdict counts per group

Rule (Algorithm 2, revised):
  E  evidence:  n+ (decision tier, calibration) >= n_min = ceil(ln d / ln(1 - a_risk))   else INSUFFICIENT
  V1 miss rate: Learn-then-Test certificate for FNR <= a_risk in the decision tier      veto
  V2 calibration: ECE_H <= 0.10                                                          veto
  V3 coverage:  one-sided 95% Clopper-Pearson lower bound of coverage >= 1 - alpha - eps  veto
  S  secondary (flag only): AURC_H <= 0.25, eta_xai >= 0.70 where computed
  verdict: INSUFFICIENT if E fails; FAIL if any veto fails; PASS-WITH-CONDITIONS if a
  secondary flag fires; PASS otherwise.
"""
import math
from pathlib import Path

import pandas as pd
from scipy.stats import beta

R = Path(__file__).resolve().parents[1] / "results"
A_RISK, DELTA, ALPHA, EPS = 0.05, 0.10, 0.10, 0.02
N_MIN = math.ceil(math.log(DELTA) / math.log(1 - A_RISK))  # 45


def cp_lower(p_hat, n, conf=0.95):
    k = round(p_hat * n)
    return 0.0 if k == 0 else float(beta.ppf(1 - conf, k, n - k + 1))


def verdict(n_pos, cert, ece_h, cov, n_test, aurc_h=None, eta=None):
    flags = []
    if n_pos is not None and n_pos < N_MIN:
        return "Insufficient", "n+<%d" % N_MIN
    if cert is not None and not bool(cert):
        flags.append("V1")
    if ece_h > 0.10:
        flags.append("V2")
    if cp_lower(cov, n_test) < 1 - ALPHA - EPS:
        flags.append("V3")
    if flags:
        return "Fail", "+".join(flags)
    sec = []
    if aurc_h is not None and aurc_h > 0.25:
        sec.append("S-AURC")
    if eta is not None and eta == eta and eta < 0.70:
        sec.append("S-XAI")
    return ("Pass-with-conditions", "+".join(sec)) if sec else ("Pass", "")


rows = []
o = pd.read_csv(R / "oracle_control.csv")
for _, r in o.iterrows():
    v, f = verdict(None, r.ltt_certified, r.ece_high, r.conformal_coverage, int(r.n_test), r.aurc_high)
    rows.append(dict(substrate="simulator", outcome="escalation", model="bayes_oracle", seed=r.seed,
                     operator="clean", level=0.0, verdict_old=r.verdict_xai_assumed_pass, verdict_new=v, flags=f))

rob = pd.read_csv(R / "synthetic_robustness_seed.csv")
ms = pd.read_csv(R / "model_seed_metrics.csv")[["seed", "model", "aurc_high"]]
st = pd.read_csv(R / "operator_stress_seed.csv")
clean = st[st.operator == "clean"][["seed", "model", "conformal_cov"]]
syn = rob.merge(clean, on=["seed", "model"], how="inner").merge(ms, on=["seed", "model"], how="left")
for _, r in syn.iterrows():
    v, f = verdict(None, r.ltt_certified_High, r.ece_high, r.conformal_cov, 1800,
                   None if pd.isna(r.aurc_high) else r.aurc_high)
    rows.append(dict(substrate="simulator", outcome="escalation", model=r.model, seed=r.seed,
                     operator="clean", level=0.0, verdict_old="", verdict_new=v, flags=f))

# perturbed inputs: no certificate exists for the perturbed regime (it was computed on clean
# calibration data), so V1 cannot be passed; V2/V3 are reported to show which vetoes fire anyway.
for _, r in st[st.operator != "clean"].iterrows():
    v, f = verdict(None, False, r.ece_high, r.conformal_cov, 1800, r.aurc_high)
    rows.append(dict(substrate="simulator", outcome="escalation", model=r.model, seed=r.seed,
                     operator=r.operator, level=r.level, verdict_old="", verdict_new=v, flags=f))

for suf, name in [("", "admission"), ("_severe", "severe"), ("_undertriage", "undertriage")]:
    k = pd.read_csv(R / f"ktas_seed_metrics{suf}.csv")
    for _, r in k.iterrows():
        v, f = verdict(int(r.ltt_n_cal_pos_high), r.ltt_certified, r.ece_High, r.conformal_coverage,
                       int(r.n_test), r.aurc_High, r.eta_occlusion)
        rows.append(dict(substrate="ktas", outcome=name, model=r.model, seed=r.seed, operator="clean",
                         level=0.0, verdict_old=r.verdict, verdict_new=v, flags=f,
                         fnr_high_nurse=r.fnr_high_nurse))

out = pd.DataFrame(rows)
out.to_csv(R / "safety_first_verdict_seed.csv", index=False)
summ = (out.groupby(["substrate", "outcome", "operator", "level", "model", "verdict_new"]).size()
        .unstack(fill_value=0).reset_index())
summ.to_csv(R / "safety_first_verdict_summary.csv", index=False)
print("n_min =", N_MIN)
print(summ.to_string())
