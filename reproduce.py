"""
Reproduce the statistics, tables and figures of
"The Gravity of Hegemony: How Much Can Nine Historical Cases Say About the
Timing of Reserve-Currency Transitions?"

Everything is computed from data/table1_cases.csv (the nine coded cases).
Closed-form or exhaustively enumerated computations are exact; the few
stochastic procedures (bootstrap, coverage experiment, rubric-error Monte
Carlo) use fixed seeds.

Usage:
    python src/reproduce.py                 # tables + checks + figures
    python src/reproduce.py --no-figures    # skip figures

Outputs go to outputs/ (CSV tables, verification.txt, regenerated figures).
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "table1_cases.csv")
OUT = os.path.join(ROOT, "outputs")

SEED = 20260101
B_BOOT = 20_000          # Section 3.5 bootstrap draws
N_COVERAGE = 3_000       # Section 3.7 coverage simulations
B_COVERAGE = 1_000       # bootstrap draws inside each coverage simulation
N_RUBRIC = 20_000        # Monte Carlo draws per flip probability
ALPHA_GRID = np.logspace(np.log10(0.05), np.log10(500), 800)   # alpha <= 500


# ----------------------------------------------------------------------------
# basic building blocks
# ----------------------------------------------------------------------------
def ols_fit(x, y):
    """OLS of y on (1, x). Returns (a, b, sse)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx < 1e-12:                       # constant regressor -> intercept only
        return ym, 0.0, ((y - ym) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    return a, b, ((y - a - b * x) ** 2).sum()


def r2(x, y):
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - ols_fit(x, y)[2] / sst


def power_x(mu, alpha):
    mu = np.asarray(mu, float)
    return np.where(mu > 0, mu ** alpha, 0.0)


def best_alpha(mu, y):
    """alpha in ALPHA_GRID maximising in-sample fit (minimising SSE)."""
    sse = [ols_fit(power_x(mu, a), y)[2] for a in ALPHA_GRID]
    return ALPHA_GRID[int(np.argmin(sse))]


def loo_predictions(mu, y, model):
    """Leave-one-out predictions. model in {'intercept','linear','step','power'}.
    For 'power', alpha is re-selected inside every fold."""
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        m, yy = mu[keep], y[keep]
        if model == "intercept":
            pred[i] = yy.mean()
        else:
            if model == "linear":
                f = lambda v: np.asarray(v, float)
            elif model == "step":
                f = lambda v: (np.asarray(v, float) == 1).astype(float)
            else:
                al = best_alpha(m, yy)
                f = lambda v, al=al: power_x(v, al)
            a, b, _ = ols_fit(f(m), yy)
            pred[i] = a + b * f(mu[i])
    return pred


def loo_r2(mu, y, model):
    p = loo_predictions(mu, y, model)
    press = ((y - p) ** 2).sum()
    return 1 - press / ((y - y.mean()) ** 2).sum(), press


def aicc(sse, n, k):
    K = k + 1
    return n * np.log(sse / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1)


def step_groups(mu, y):
    full = mu == 1
    return y[full], y[~full]


def perm_p(labels_mask, y, stat="absdiff"):
    """Exact permutation p-value for difference in means: enumerate all
    assignments of k 'full' labels to n cases."""
    n, k = len(y), int(labels_mask.sum())
    obs = abs(y[labels_mask].mean() - y[~labels_mask].mean())
    count, total = 0, 0
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        d = abs(y[m].mean() - y[~m].mean())
        total += 1
        if d >= obs - 1e-9:
            count += 1
    return count / total, count, total


def pooled_pi(y_full, y_part, level=0.95, group_var=False):
    """Prediction interval for a new full-sovereignty issuer (Prop. 3.5)."""
    n1 = len(y_full)
    m1 = y_full.mean()
    if group_var:
        nu = n1 - 1
        s = y_full.std(ddof=1)
    else:
        nu = len(y_full) + len(y_part) - 2
        sse = ((y_full - m1) ** 2).sum() + ((y_part - y_part.mean()) ** 2).sum()
        s = np.sqrt(sse / nu)
    h = stats.t.ppf(0.5 + level / 2, nu) * s * np.sqrt(1 + 1 / n1)
    return m1 - h, m1 + h, nu, s


def step_stats(mu, y):
    """step-model R^2, group means, gap."""
    full = mu == 1
    if full.all() or (~full).all():
        return 0.0, y.mean(), y.mean()
    x = full.astype(float)
    return r2(x, y), y[full].mean(), y[~full].mean()


# ----------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------
def main(make_figures=True):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(DATA)
    mu = d["mu"].to_numpy(float)
    y = d["tau_mid"].to_numpy(float)
    n = len(y)
    pre1900 = (d["crossing_year"] < 1900).to_numpy()
    full = mu == 1
    sst = ((y - y.mean()) ** 2).sum()
    R = {}                  # computed results
    tables = {}

    # ---- Section 5.1 / Table 2: model comparison -----------------------------
    R["R2_profile_alpha0.1"] = r2(power_x(mu, 0.1), y)
    R["R2_profile_alpha1"] = r2(power_x(mu, 1.0), y)
    a_best = best_alpha(mu, y)
    R["R2_profile_best_alpha"] = r2(power_x(mu, a_best), y)
    R["R2_step"] = r2(full.astype(float), y)
    rows = []
    for name, k, x in [("Intercept only", 1, np.zeros(n)),
                       ("Linear in mu", 2, mu),
                       ("Power curve (best alpha)", 3, power_x(mu, a_best)),
                       ("Step at mu=1", 2, full.astype(float))]:
        sse = ols_fit(x, y)[2]
        key = {"Intercept only": "intercept", "Linear in mu": "linear",
               "Power curve (best alpha)": "power", "Step at mu=1": "step"}[name]
        l2, press = loo_r2(mu, y, key)
        rows.append({"Model": name, "k": k, "In-sample R2": 1 - sse / sst,
                     "LOO R2": l2, "LOO RMSE": np.sqrt(press / n),
                     "AICc": aicc(sse, n, k)})
    tables["table2_model_comparison"] = pd.DataFrame(rows)
    t2 = tables["table2_model_comparison"].set_index("Model")
    R["LOO_R2_intercept"] = t2.loc["Intercept only", "LOO R2"]
    R["LOO_R2_linear"] = t2.loc["Linear in mu", "LOO R2"]
    R["LOO_R2_power"] = t2.loc["Power curve (best alpha)", "LOO R2"]
    R["LOO_R2_step"] = t2.loc["Step at mu=1", "LOO R2"]
    R["AICc_step"] = t2.loc["Step at mu=1", "AICc"]
    R["AICc_linear"] = t2.loc["Linear in mu", "AICc"]
    R["AICc_power"] = t2.loc["Power curve (best alpha)", "AICc"]
    R["AICc_intercept"] = t2.loc["Intercept only", "AICc"]

    yf, yp = step_groups(mu, y)
    R["mean_full"], R["mean_partial"] = yf.mean(), yp.mean()
    R["gap"] = yf.mean() - yp.mean()

    # ---- Section 5.3: intervals ---------------------------------------------
    lo, hi, nu, s = pooled_pi(yf, yp)
    R["PI_pooled_lo"], R["PI_pooled_hi"], R["sigma_hat"] = lo, hi, s
    lo_g, hi_g, _, _ = pooled_pi(yf, yp, group_var=True)
    R["PI_group_lo"], R["PI_group_hi"] = lo_g, hi_g

    idx = rng.integers(0, n, size=(B_BOOT, n))
    has_full = (full[idx]).any(axis=1)
    R["boot_discarded"] = int((~has_full).sum())
    idx = idx[has_full]
    fm = np.where(full[idx], y[idx], np.nan)
    boot_means = np.nanmean(fm, axis=1)
    R["boot_lo"], R["boot_hi"] = np.percentile(boot_means, [2.5, 97.5])
    R["boot_median"] = np.median(boot_means)

    # log-scale predictive interval (positivity)
    ly = np.log(y)
    lyf, lyp = step_groups(mu, ly)
    llo, lhi, _, _ = pooled_pi(lyf, lyp)
    R["PI_log_lo"], R["PI_log_hi"] = np.exp(llo), np.exp(lhi)

    # verified subsample (drop benchmark-only cases)
    ver = (d["provenance"] != "BO").to_numpy()
    mv, yv = mu[ver], y[ver]
    yvf, yvp = step_groups(mv, yv)
    vlo, vhi, _, _ = pooled_pi(yvf, yvp)
    vglo, vghi, _, _ = pooled_pi(yvf, yvp, group_var=True)
    R["ver_PI_pooled_lo"], R["ver_PI_pooled_hi"] = vlo, vhi
    R["ver_PI_group_halfwidth"] = (vghi - vglo) / 2
    vidx = rng.integers(0, len(yv), size=(B_BOOT, len(yv)))
    vfull = mv == 1
    vok = vfull[vidx].any(axis=1)
    vm = np.nanmean(np.where(vfull[vidx[vok]], yv[vidx[vok]], np.nan), axis=1)
    R["ver_boot_lo"], R["ver_boot_hi"] = np.percentile(vm, [2.5, 97.5])

    tables["section5_3_intervals"] = pd.DataFrame([
        ["Bootstrap CI of the mean (pairs, B=20,000)", R["boot_lo"], R["boot_hi"]],
        ["Predictive, pooled variance", lo, hi],
        ["Predictive, group variance", lo_g, hi_g],
        ["Predictive, pooled variance, log scale", R["PI_log_lo"], R["PI_log_hi"]],
        ["Verified subsample: bootstrap CI", R["ver_boot_lo"], R["ver_boot_hi"]],
        ["Verified subsample: predictive (pooled)", vlo, vhi],
    ], columns=["interval", "lower", "upper"])

    # ---- Section 3.6 / 5.4: exact tests -------------------------------------
    p_perm, c_perm, tot = perm_p(full, y)
    R["perm_p"], R["perm_count"], R["perm_total"] = p_perm, c_perm, tot
    R["mannwhitney_p"] = stats.mannwhitneyu(yf, yp, alternative="two-sided",
                                            method="exact").pvalue
    R["fisher_p"] = stats.fisher_exact([[4, 0], [0, 5]])[1]
    R["welch_p"] = stats.ttest_ind(yf, yp, equal_var=False).pvalue
    sp = np.sqrt(((len(yf) - 1) * yf.var(ddof=1) + (len(yp) - 1) * yp.var(ddof=1))
                 / (len(yf) + len(yp) - 2))
    R["cohen_d"] = (yf.mean() - yp.mean()) / sp
    R["hedges_g"] = R["cohen_d"] * (1 - 3 / (4 * (n - 2) - 1))
    R["ver_perm_p"] = perm_p(vfull, yv)[0]
    R["ver_mannwhitney_p"] = stats.mannwhitneyu(
        yvf, yvp, alternative="two-sided", method="exact").pvalue

    # ---- Table 4: leave-one-case-out -----------------------------------------
    rows = []
    for i in range(n):
        keep = np.arange(n) != i
        m, yy = mu[keep], y[keep]
        f = m == 1
        r2_, mf, mp = step_stats(m, yy)
        pp = perm_p(f, yy)[0]
        rows.append({"Case dropped": d.loc[i, "case"], "n_mu=1": int(f.sum()),
                     "mean mu=1": mf, "mean mu<1": mp, "Gap": mf - mp,
                     "Perm p": pp, "Step R2": r2_})
    t4 = pd.DataFrame(rows)
    tables["table4_leave_one_case_out"] = t4
    R["t4_gap_min"], R["t4_gap_max"] = t4["Gap"].min(), t4["Gap"].max()
    R["t4_p_min"], R["t4_p_max"] = t4["Perm p"].min(), t4["Perm p"].max()

    # ---- Table 5: sovereignty vs era partitions -----------------------------
    era_full = pre1900
    rows = []
    for name, mask in [("Sovereignty (mu=1 vs mu<1)", full),
                       ("Era (crossing before vs after 1900)", era_full)]:
        g1, g2 = y[mask], y[~mask]
        l2, _ = loo_r2(mask.astype(float), y, "linear")   # two-group = linear in indicator
        rows.append({"Partition": name, "Sizes": f"{mask.sum()}/{(~mask).sum()}",
                     "Mean lag 1": g1.mean(), "Mean lag 2": g2.mean(),
                     "Gap": g1.mean() - g2.mean(),
                     "Perm p": perm_p(mask, y)[0],
                     "Step R2": r2(mask.astype(float), y), "LOO R2": l2})
    t5 = pd.DataFrame(rows)
    tables["table5_partitions"] = t5
    R["era_perm_p"] = t5.loc[1, "Perm p"]
    R["era_R2"] = t5.loc[1, "Step R2"]
    R["era_LOO_R2"] = t5.loc[1, "LOO R2"]
    R["era_mean_pre"], R["era_mean_post"] = t5.loc[1, "Mean lag 1"], t5.loc[1, "Mean lag 2"]
    R["corr_indicators"] = np.corrcoef(full.astype(float), era_full.astype(float))[0, 1]
    X = np.column_stack([np.ones(n), full.astype(float), era_full.astype(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    R["R2_both_indicators"] = 1 - ((y - X @ beta) ** 2).sum() / sst

    # outcome-contaminated scoring
    def rescored(idx_list):
        m2 = mu.copy()
        m2[idx_list] = 1.0
        f2 = m2 == 1
        a, b = y[f2], y[~f2]
        l_, h_, _, _ = pooled_pi(a, b)
        return a.mean(), b.mean(), r2(f2.astype(float), y), l_, h_
    ott, yen = list(d["case"]).index("Ottoman lira"), list(d["case"]).index("Imperial yen")
    R["rescore_ott"] = rescored([ott])
    R["rescore_ott_yen"] = rescored([ott, yen])

    # ---- Table 6: leave-one-out predictions ----------------------------------
    ps = loo_predictions(mu, y, "step")
    pp = loo_predictions(mu, y, "power")
    t6 = pd.DataFrame({"Case": d["case"], "mu": mu, "tau (mid)": y,
                       "Step LOO": ps, "Power LOO": pp, "Step error": y - ps})
    tables["table6_loo_predictions"] = t6
    R["max_step_loo_error"] = np.abs(y - ps).max()

    # ---- Section 5.5: sensitivity to coding choices ---------------------------
    taus = d[["tau_narrow", "tau_mid", "tau_wide"]].to_numpy(float)
    means_full, r2s, pi_narrow, pi_wide = [], [], None, None
    for combo in itertools.product(range(3), repeat=n):
        yy = taus[np.arange(n), combo]
        a, b = yy[full], yy[~full]
        means_full.append(a.mean())
        r2s.append(r2(full.astype(float), yy))
    means_full, r2s = np.array(means_full), np.array(r2s)
    R["n_date_combos"] = len(means_full)
    R["datecombo_mean_min"], R["datecombo_mean_max"] = means_full.min(), means_full.max()
    R["datecombo_R2_min"], R["datecombo_R2_max"] = r2s.min(), r2s.max()
    R["datecombo_R2_median"] = np.median(r2s)
    yn, yw = taus[:, 0], taus[:, 2]
    R["narrow_only_mean"], R["wide_only_mean"] = yn[full].mean(), yw[full].mean()
    pn = pooled_pi(yn[full], yn[~full])
    pw = pooled_pi(yw[full], yw[~full])
    R["narrow_PI_lo"], R["narrow_PI_hi"] = pn[0], pn[1]
    R["wide_PI_lo"], R["wide_PI_hi"] = pw[0], pw[1]

    # rubric errors (Monte Carlo)
    rub = d[["E", "I", "D", "C"]].to_numpy(int)          # n x 4
    rub_rows = []
    flip_r2_p10 = None
    for p in (0.05, 0.10, 0.20):
        flips = rng.random((N_RUBRIC, n, 4)) < p
        new = np.where(flips, 1 - rub[None, :, :], rub[None, :, :])
        new_full = new.sum(axis=2) == 4                    # mu == 1  <=> all four met
        r2_draws = np.empty(N_RUBRIC)
        for j in range(N_RUBRIC):
            f = new_full[j]
            r2_draws[j] = 0.0 if (f.all() or (~f).all()) else r2(f.astype(float), y)
        rub_rows.append({"p": p, "median R2": np.median(r2_draws),
                         "5th percentile R2": np.percentile(r2_draws, 5)})
        if p == 0.10:
            flip_r2_p10 = r2_draws
    t_rub = pd.DataFrame(rub_rows)
    tables["section5_5_rubric_errors"] = t_rub
    R["rub_med_05"], R["rub_med_10"], R["rub_med_20"] = t_rub["median R2"]
    R["rub_p5_05"], R["rub_p5_10"] = t_rub["5th percentile R2"][:2]

    # ---- Section 3.7 / Table 3: coverage experiment ---------------------------
    sigma, m_full_true, m_part_true = 11.0, 52.0, 14.0
    mu_true = np.where(full, m_full_true, m_part_true)
    cover = {"boot": 0, "pooled": 0, "group": 0}
    length = {"boot": 0.0, "pooled": 0.0, "group": 0.0}
    for _ in range(N_COVERAGE):
        ysim = mu_true + rng.normal(0, sigma, n)
        new = m_full_true + rng.normal(0, sigma)
        a, b = ysim[full], ysim[~full]
        # bootstrap CI of the mean at mu=1
        ii = rng.integers(0, n, size=(B_COVERAGE, n))
        ok = full[ii].any(axis=1)
        bm = np.nanmean(np.where(full[ii[ok]], ysim[ii[ok]], np.nan), axis=1)
        blo, bhi = np.percentile(bm, [2.5, 97.5])
        plo, phi, _, _ = pooled_pi(a, b)
        glo, ghi, _, _ = pooled_pi(a, b, group_var=True)
        for k, (l_, h_) in {"boot": (blo, bhi), "pooled": (plo, phi),
                            "group": (glo, ghi)}.items():
            cover[k] += (l_ <= new <= h_)
            length[k] += h_ - l_
    t3 = pd.DataFrame([{"Interval": k, "Coverage": cover[k] / N_COVERAGE,
                        "Mean length (yrs)": length[k] / N_COVERAGE}
                       for k in ("boot", "pooled", "group")])
    tables["table3_coverage"] = t3
    R["cov_boot"], R["cov_pooled"], R["cov_group"] = t3["Coverage"]
    R["len_boot"], R["len_pooled"], R["len_group"] = t3["Mean length (yrs)"]
    # closed form (Proposition 3.6): coverage of a known-sigma 95% CI as a predictive interval
    R["prop36_n4"] = 2 * stats.norm.cdf(1.96 / np.sqrt(4 + 1)) - 1
    R["prop36_n2"] = 2 * stats.norm.cdf(1.96 / np.sqrt(2 + 1)) - 1

    # ---- write tables + verification ------------------------------------------
    for name, df in tables.items():
        df.round(4).to_csv(os.path.join(OUT, name + ".csv"), index=False)

    checks = [
        # (label, computed, reported in paper, tolerance)
        ("R2 profile, alpha=0.1", R["R2_profile_alpha0.1"], 0.42, 0.01),
        ("R2 profile, alpha=1 (linear)", R["R2_profile_alpha1"], 0.695, 0.001),
        ("R2 step / profile limit", R["R2_step"], 0.704, 0.001),
        ("tau_max (mean, mu=1)", R["mean_full"], 52.0, 0.05),
        ("tau_min (mean, mu<1)", R["mean_partial"], 14.2, 0.05),
        ("Group difference (years)", R["gap"], 37.8, 0.05),
        ("LOO R2 intercept", R["LOO_R2_intercept"], -0.27, 0.01),
        ("LOO R2 linear", R["LOO_R2_linear"], 0.54, 0.01),
        ("LOO R2 power (alpha re-selected)", R["LOO_R2_power"], 0.35, 0.03),
        ("LOO R2 step", R["LOO_R2_step"], 0.52, 0.01),
        ("AICc intercept", R["AICc_intercept"], 61.9, 0.1),
        ("AICc linear", R["AICc_linear"], 56.1, 0.1),
        ("AICc power", R["AICc_power"], 63.0, 0.1),
        ("AICc step", R["AICc_step"], 55.8, 0.1),
        ("Pooled residual s.d.", R["sigma_hat"], 13.8, 0.05),
        ("Predictive interval, lower (pooled)", R["PI_pooled_lo"], 15.5, 0.1),
        ("Predictive interval, upper (pooled)", R["PI_pooled_hi"], 88.5, 0.1),
        ("Predictive interval, lower (group var.)", R["PI_group_lo"], 12.1, 0.1),
        ("Predictive interval, upper (group var.)", R["PI_group_hi"], 91.9, 0.1),
        ("Bootstrap CI lower (seeded; may differ)", R["boot_lo"], 41.2, 1.0),
        ("Bootstrap CI upper (seeded; may differ)", R["boot_hi"], 62.2, 1.0),
        ("Bootstrap median (seeded; may differ)", R["boot_median"], 52.0, 1.0),
        ("Verified subsample: bootstrap lower", R["ver_boot_lo"], 45.0, 0.01),
        ("Verified subsample: bootstrap upper", R["ver_boot_hi"], 63.0, 0.01),
        ("Verified subsample: pooled PI lower", R["ver_PI_pooled_lo"], 6.9, 0.1),
        ("Verified subsample: pooled PI upper", R["ver_PI_pooled_hi"], 101.1, 0.1),
        ("Verified subsample: group-variance half-width", R["ver_PI_group_halfwidth"], 198, 1.0),
        ("Log-scale PI lower (full sample)", R["PI_log_lo"], 7.1, 0.2),
        ("Log-scale PI upper (full sample)", R["PI_log_hi"], 366, 5.0),
        ("Exact permutation p", R["perm_p"], 0.0238, 0.0001),
        ("Permutations at least as extreme", R["perm_count"], 3, 0),
        ("Exact Mann-Whitney p", R["mannwhitney_p"], 0.032, 0.001),
        ("Fisher exact p (reserve status)", R["fisher_p"], 0.0079, 0.0001),
        ("Welch t-test p", R["welch_p"], 0.004, 0.001),
        ("Cohen's d", R["cohen_d"], 2.74, 0.01),
        ("Hedges' g", R["hedges_g"], 2.44, 0.01),
        ("Verified subsample permutation p", R["ver_perm_p"], 0.048, 0.001),
        ("Verified subsample Mann-Whitney p", R["ver_mannwhitney_p"], 0.095, 0.001),
        ("Table 4: min gap", R["t4_gap_min"], 34.1, 0.05),
        ("Table 4: max gap", R["t4_gap_max"], 44.5, 0.05),
        ("Table 4: min perm p", R["t4_p_min"], 0.018, 0.001),
        ("Table 4: max perm p", R["t4_p_max"], 0.057, 0.001),
        ("Era partition: perm p", R["era_perm_p"], 0.0079, 0.0001),
        ("Era partition: step R2", R["era_R2"], 0.882, 0.001),
        ("Era partition: LOO R2", R["era_LOO_R2"], 0.81, 0.01),
        ("Era partition: mean lag before 1900", R["era_mean_pre"], 49.8, 0.05),
        ("Era partition: mean lag after 1900", R["era_mean_post"], 7.5, 0.05),
        ("Correlation of the two indicators", R["corr_indicators"], 0.80, 0.01),
        ("R2 with both indicators", R["R2_both_indicators"], 0.904, 0.001),
        ("Re-score Ottoman: mean mu=1", R["rescore_ott"][0], 49.8, 0.05),
        ("Re-score Ottoman: mean mu<1", R["rescore_ott"][1], 7.5, 0.05),
        ("Re-score Ottoman: step R2", R["rescore_ott"][2], 0.88, 0.005),
        ("Re-score Ottoman: PI lower", R["rescore_ott"][3], 27.2, 0.1),
        ("Re-score Ottoman: PI upper", R["rescore_ott"][4], 72.4, 0.1),
        ("Re-score lira+yen: mean mu=1", R["rescore_ott_yen"][0], 43.7, 0.05),
        ("Re-score lira+yen: step R2", R["rescore_ott_yen"][2], 0.64, 0.005),
        ("Re-score lira+yen: PI lower", R["rescore_ott_yen"][3], 4.8, 0.1),
        ("Re-score lira+yen: PI upper", R["rescore_ott_yen"][4], 82.5, 0.1),
        ("Largest step LOO error (years)", R["max_step_loo_error"], 33.5, 0.05),
        ("Date combinations enumerated", R["n_date_combos"], 19683, 0),
        ("Date combos: min mu=1 mean", R["datecombo_mean_min"], 45.0, 0.01),
        ("Date combos: max mu=1 mean", R["datecombo_mean_max"], 58.75, 0.01),
        ("Date combos: min step R2", R["datecombo_R2_min"], 0.57, 0.01),
        ("Date combos: max step R2", R["datecombo_R2_max"], 0.82, 0.01),
        ("Date combos: median step R2", R["datecombo_R2_median"], 0.69, 0.01),
        ("Narrow-only pooled PI lower", R["narrow_PI_lo"], 14.6, 0.1),
        ("Narrow-only pooled PI upper", R["narrow_PI_hi"], 75.4, 0.1),
        ("Wide-only pooled PI lower", R["wide_PI_lo"], 17.3, 0.1),
        ("Wide-only pooled PI upper", R["wide_PI_hi"], 100.2, 0.1),
        ("Rubric errors p=0.05: median R2 (MC)", R["rub_med_05"], 0.62, 0.05),
        ("Rubric errors p=0.10: median R2 (MC)", R["rub_med_10"], 0.34, 0.05),
        ("Rubric errors p=0.20: median R2 (MC)", R["rub_med_20"], 0.24, 0.05),
        ("Coverage: bootstrap CI (MC)", R["cov_boot"], 0.52, 0.04),
        ("Coverage: predictive, pooled (MC)", R["cov_pooled"], 0.95, 0.02),
        ("Coverage: predictive, group (MC)", R["cov_group"], 0.95, 0.02),
        ("Mean length: bootstrap CI (MC)", R["len_boot"], 19, 3),
        ("Mean length: pooled PI (MC)", R["len_pooled"], 56, 4),
        ("Mean length: group PI (MC)", R["len_group"], 71, 6),
        ("Prop. 3.6 coverage, n1=4", R["prop36_n4"], 0.62, 0.005),
        ("Prop. 3.6 coverage, n1=2", R["prop36_n2"], 0.74, 0.005),
    ]
    lines = ["VERIFICATION: recomputed values vs. values reported in the paper",
             "=" * 78,
             f"{'quantity':<50}{'computed':>11}{'paper':>10}  status", "-" * 78]
    bad = 0
    for label, comp, rep, tol in checks:
        ok = abs(comp - rep) <= tol + 1e-12
        bad += (not ok)
        lines.append(f"{label:<50}{comp:>11.4f}{rep:>10.4f}  {'ok' if ok else 'DIFFERS'}")
    lines += ["-" * 78, f"{len(checks) - bad}/{len(checks)} quantities match "
              f"(tolerances reflect the precision printed in the paper; seeded "
              f"Monte Carlo rows can differ slightly)."]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "verification.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nTables written to", OUT)

    if make_figures:
        make_figs(d, mu, y, full, taus, means_full, flip_r2_p10, R, ps, pp)
    return R, tables


# ----------------------------------------------------------------------------
# figures (regenerated; styling differs slightly from the paper's PNGs)
# ----------------------------------------------------------------------------
def make_figs(d, mu, y, full, taus, means_full, flip_r2, R, ps, pp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: lag against mu
    fig, ax = plt.subplots(figsize=(7, 4.4))
    xs = np.linspace(0, 1, 200)
    a1, b1, _ = ols_fit(mu, y)
    ax.plot(xs, a1 + b1 * xs, "--", color="grey", label="linear (alpha=1)")
    for al, st in [(2, ":")]:
        a_, b_, _ = ols_fit(mu ** al, y)
        ax.plot(xs, a_ + b_ * xs ** al, st, color="grey", label=f"power curve, alpha={al}")
    ax.hlines(R["mean_partial"], 0, 1, color="black", lw=1.8, label="step fit")
    ax.hlines(R["mean_full"], 0.95, 1.05, color="black", lw=1.8)
    for i in range(len(d)):
        bo = d.loc[i, "provenance"] == "BO"
        ax.vlines(mu[i], taus[i, 0], taus[i, 2], color="black", lw=0.8)
        ax.plot(mu[i], y[i], "s" if bo else "o", mfc="black" if full[i] else "white",
                mec="black", ms=6)
        ax.annotate(d.loc[i, "case"], (mu[i], y[i]), textcoords="offset points",
                    xytext=(6, -3), fontsize=7)
    ax.set_xlabel("Monetary Sovereignty Modifier mu")
    ax.set_ylabel("Transition lag tau (years)")
    ax.set_xlim(-0.05, 1.25)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_scatter.png"), dpi=200)
    plt.close(fig)

    # Figure 2: exact profile of R^2 over alpha
    grid = ALPHA_GRID[::8]
    ins = [r2(power_x(mu, a), y) for a in grid]
    loo = []
    for a in grid:
        p = np.empty(len(y))
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            aa, bb, _ = ols_fit(power_x(mu[keep], a), y[keep])
            p[i] = aa + bb * power_x(mu[i], a)
        loo.append(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.semilogx(grid, ins, "-", color="black", label="in-sample R2 (exact profile)")
    ax.semilogx(grid, loo, "--", color="black", label="leave-one-out R2 at fixed alpha")
    ax.axhline(R["R2_step"], ls=":", color="grey", label="step-function limit")
    ax.set_xlabel("shape parameter alpha (log scale)")
    ax.set_ylabel("R2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_profile.png"), dpi=200)
    plt.close(fig)

    # Figure 3: intervals
    items = [("Bootstrap CI of mean", R["boot_lo"], R["boot_hi"], "solid"),
             ("Predictive, pooled variance", R["PI_pooled_lo"], R["PI_pooled_hi"], "hatch"),
             ("Predictive, group variance", R["PI_group_lo"], R["PI_group_hi"], "hatch"),
             ("Verified subsample: bootstrap CI", R["ver_boot_lo"], R["ver_boot_hi"], "solid"),
             ("Verified subsample: predictive (pooled)", R["ver_PI_pooled_lo"],
              R["ver_PI_pooled_hi"], "hatch")]
    fig, ax = plt.subplots(figsize=(8, 3))
    for j, (lab, lo, hi, kind) in enumerate(items[::-1]):
        ax.barh(j, hi - lo, left=lo, height=0.5, color="black" if kind == "solid" else "white",
                edgecolor="black", hatch=None if kind == "solid" else "////")
        ax.text(hi + 1.5, j, f"[{lo:.0f}, {hi:.0f}]", va="center", fontsize=7)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([i[0] for i in items[::-1]], fontsize=7)
    ax.axvline(R["mean_full"], ls=":", color="grey")
    ax.set_xlabel("95% interval for tau of a mu=1 issuer (years)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_intervals.png"), dpi=200)
    plt.close(fig)

    # Figure 4: robustness
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(means_full, bins=30, color="grey", edgecolor="black")
    axes[0].set_xlabel("mean tau for mu=1 over all 3^9 date definitions")
    axes[0].set_ylabel("count")
    axes[1].hist(flip_r2, bins=30, color="grey", edgecolor="black")
    axes[1].set_xlabel("step-fit R2 when each rubric score is flipped w.p. 0.10")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_robustness.png"), dpi=200)
    plt.close(fig)
    print("Figures written to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    main(make_figures=not args.no_figures)
