"""Generate every data table of the paper and supplement as Markdown.

Usage (from the repository root):

    python scripts/make_tables.py

Writes tables/<name>.md (T3-T8 for the paper, S_* for the supplement) and
tables/stats.csv with the divergence test statistics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--results", default="results/all_results.csv")
ap.add_argument("--divergence", default="results/divergence_all_datasets.csv")
ap.add_argument("--out", default="tables")
args = ap.parse_args()
df = pd.read_csv(args.results)
dv = pd.read_csv(args.divergence)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
REAL = ["mean", "ffill", "knn", "mice", "usgan", "saits", "csdi"]
ORDER = ["no_repair", "mean", "ffill", "knn", "mice", "usgan", "saits", "csdi", "oracle_clean"]
DS = ["FD001", "FD002", "FD003", "FD004"]
NM = {
    "mean": "Mean",
    "ffill": "Forward-fill",
    "knn": "KNN",
    "mice": "MICE†",
    "usgan": "US-GAN",
    "saits": "SAITS",
    "csdi": "CSDI†",
    "oracle_clean": "Clean input (ref.)",
    "no_repair": "No repair (ref.)",
}


def w(name, text):
    (OUT / f"{name}.md").write_text(text.strip() + "\n")


def fmt(v, d):
    return "–" if pd.isna(v) else f"{v:,.{d}f}"


g = df.groupby(["imputer", "dataset"])

# ---- T3 reference bounds ----
rows = [
    "| Subset | Train windows | Test engines | Clean RUL RMSE | Clean PHM08 "
    "| Clean coverage | Clean risk-aware regret | No-repair RUL RMSE "
    "| No-repair PHM08 | No-repair risk-aware regret |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
sizes = {"FD001": (17731, 100), "FD002": (46219, 259), "FD003": (21820, 100), "FD004": (54028, 248)}
for ds in DS:
    c = df[(df.dataset == ds) & (df.imputer == "oracle_clean")].mean(numeric_only=True)
    n = df[(df.dataset == ds) & (df.imputer == "no_repair")].mean(numeric_only=True)
    rows.append(
        f"| {ds} | {sizes[ds][0]:,} | {sizes[ds][1]} | {c.rul_rmse:.2f}"
        f" | {c.phm08:,.0f} | {c.coverage:.2f} | {c.regret_risk:.1f}"
        f" | {n.rul_rmse:.2f} | {n.phm08:,.0f} | {n.regret_risk:.1f} |"
    )
w("T3", "\n".join(rows))

# ---- T4 imputation RMSE and RUL RMSE ----
m = g[["imp_rmse", "rul_rmse"]].mean()
hdr = (
    "| Method | "
    + " | ".join(f"Imp. RMSE {d}" for d in DS)
    + " | "
    + " | ".join(f"RUL RMSE {d}" for d in DS)
    + " |"
)
rows = [hdr, "|---" * 9 + "|"]
best_imp = {d: m.xs(d, level=1).loc[REAL].imp_rmse.idxmin() for d in DS}
best_rul = {d: m.xs(d, level=1).loc[REAL].rul_rmse.idxmin() for d in DS}
for imp in ORDER:
    cells = []
    for d in DS:
        v = fmt(m.loc[(imp, d), "imp_rmse"], 3)
        cells.append(f"**{v}**" if best_imp[d] == imp else v)
    for d in DS:
        v = fmt(m.loc[(imp, d), "rul_rmse"], 2)
        cells.append(f"**{v}**" if best_rul[d] == imp else v)
    rows.append(f"| {NM[imp]} | " + " | ".join(cells) + " |")
w("T4", "\n".join(rows))

# ---- T5 two-layer divergence ----
rows = [
    "| Subset | τ_K(imp. RMSE, RUL RMSE) | τ_K(imp. RMSE, PHM08) "
    "| τ_K(imp. RMSE, point regret) [cell range] "
    "| τ_K(imp. RMSE, risk-aware regret) | Cells with accuracy τ_K > "
    "decision τ_K |",
    "|---|---|---|---|---|---|",
]
allg = []
for ds in DS:
    r = []
    for _, gg in df[(df.dataset == ds) & df.imputer.isin(REAL)].groupby(["mech", "rate"]):
        a = gg.groupby("imputer")[["imp_rmse", "rul_rmse", "phm08", "regret", "regret_risk"]].mean()
        r.append(
            [
                kendalltau(a.imp_rmse, a[c])[0]
                for c in ["rul_rmse", "phm08", "regret", "regret_risk"]
            ]
        )
    r = np.array(r)
    allg.append(r)
    pos = int((r[:, 1] > r[:, 2]).sum())
    rows.append(
        f"| {ds} | {r[:, 0].mean():.2f} | {r[:, 1].mean():.2f} | "
        f"{r[:, 2].mean():.2f} [{r[:, 2].min():.2f}, "
        f"{r[:, 2].max():.2f}] | {r[:, 3].mean():.2f} | {pos}/9 |"
    )
allg = np.vstack(allg)
gap_phm = allg[:, 1] - allg[:, 2]
gap_rul = allg[:, 0] - allg[:, 2]
p1 = wilcoxon(gap_phm, alternative="greater").pvalue
p2 = wilcoxon(gap_rul, alternative="greater").pvalue
rows.append(
    f"| All 36 cells | {allg[:, 0].mean():.2f} | {allg[:, 1].mean():.2f}"
    f" | {allg[:, 2].mean():.2f} | {allg[:, 3].mean():.2f} | "
    f"{int((gap_phm > 0).sum())}/36 |"
)
w("T5", "\n".join(rows))
stats = {
    "gap_phm_mean": gap_phm.mean(),
    "gap_phm_pos": int((gap_phm > 0).sum()),
    "p_phm": p1,
    "gap_rul_pos": int((gap_rul > 0).sum()),
    "p_rul": p2,
}

# ---- T6 point vs risk-aware regret ----
m = g[["regret", "regret_risk"]].mean()
rows = [
    "| Method | " + " | ".join(f"{d} point / risk-aware (ratio)" for d in DS) + " |",
    "|---" * 5 + "|",
]
best = {d: m.xs(d, level=1).loc[REAL].regret_risk.idxmin() for d in DS}
for imp in ORDER:
    cells = []
    for d in DS:
        a, b = m.loc[(imp, d)]
        rr = f"**{b:.1f}**" if best[d] == imp else f"{b:.1f}"
        cells.append(f"{a:.1f} / {rr} ({a / b:.2f}×)")
    rows.append(f"| {NM[imp]} | " + " | ".join(cells) + " |")
w("T6", "\n".join(rows))

# ---- T7 B ablation ----
mi = (
    df[df.imputer.isin(["csdi", "mice"])]
    .groupby(["dataset", "imputer"])[
        ["coverage_noB", "coverage", "width_noB", "width", "regret_risk_noB", "regret_risk"]
    ]
    .mean()
)
rows = [
    "| Subset | Method | Coverage w/o B | Coverage with B | Width w/o B "
    "| Width with B | Risk-aware regret w/o B | Risk-aware regret with B "
    "| Δ regret from B |",
    "|---" * 9 + "|",
]
for (ds, imp), r in mi.iterrows():
    rows.append(
        f"| {ds} | {NM[imp].rstrip('†')} | {r.coverage_noB:.3f} | "
        f"{r.coverage:.3f} | {r.width_noB:.1f} | {r.width:.1f} | "
        f"{r.regret_risk_noB:.1f} | {r.regret_risk:.1f} | "
        f"{r.regret_risk - r.regret_risk_noB:+.1f} |"
    )
w("T7", "\n".join(rows))

# ---- T8 selection criterion ----
rows = [
    "| Subset | In-sample: RMSE pick → cost | In-sample: decision pick → "
    "cost | Out-of-sample: RMSE pick → cost | Out-of-sample: decision pick"
    " → cost | Out-of-sample saving | Per-cell selection: saving (better/"
    "worse cells) |",
    "|---" * 7 + "|",
]
for ds in DS:
    d = df[(df.dataset == ds) & df.imputer.isin(REAL)]
    a = d.groupby("imputer")[["imp_rmse", "regret_risk"]].mean()
    ir, it = a.imp_rmse.idxmin(), a.regret_risk.idxmin()
    sel = d[d.seed < 5].groupby("imputer")[["imp_rmse", "regret_risk"]].mean()
    ev = d[d.seed >= 5].groupby("imputer").regret_risk.mean()
    orr, ot = sel.imp_rmse.idxmin(), sel.regret_risk.idxmin()
    tr = tt = 0.0
    better = worse = 0
    for _, gg in d.groupby(["mech", "rate"]):
        s = gg[gg.seed < 5].groupby("imputer")[["imp_rmse", "regret_risk"]].mean()
        e = gg[gg.seed >= 5].groupby("imputer").regret_risk.mean()
        cr, ct = e[s.imp_rmse.idxmin()], e[s.regret_risk.idxmin()]
        tr += cr
        tt += ct
        better += ct < cr
        worse += ct > cr
    rows.append(
        f"| {ds} | {NM[ir].rstrip('†')} → {a.loc[ir, 'regret_risk']:.2f} | "
        f"{NM[it].rstrip('†')} → {a.loc[it, 'regret_risk']:.2f} | "
        f"{NM[orr].rstrip('†')} → {ev[orr]:.2f} | "
        f"{NM[ot].rstrip('†')} → {ev[ot]:.2f} | "
        f"{100 * (1 - ev[ot] / ev[orr]):.1f}% | "
        f"{100 * (1 - tt / tr):+.1f}% ({better}/{worse}) |"
    )
w("T8", "\n".join(rows))

# ================= supplementary =================
z = 1.96
for ds in DS:
    d = df[df.dataset == ds]
    rows = [
        "| Method | Imp. RMSE | RUL RMSE | PHM08 | Coverage | Width "
        "| Point regret | Risk-aware regret |",
        "|---" * 8 + "|",
    ]
    for imp in ORDER:
        s = (
            d[d.imputer == imp]
            .groupby("seed")[
                ["imp_rmse", "rul_rmse", "phm08", "coverage", "width", "regret", "regret_risk"]
            ]
            .mean()
        )
        cells = []
        for c, dd in [
            ("imp_rmse", 3),
            ("rul_rmse", 2),
            ("phm08", 0),
            ("coverage", 3),
            ("width", 1),
            ("regret", 1),
            ("regret_risk", 1),
        ]:
            mu, sd = s[c].mean(), s[c].std(ddof=1)
            ci = z * sd / np.sqrt(len(s))
            cells.append("–" if pd.isna(mu) else f"{mu:,.{dd}f} ± {ci:,.{dd}f}")
        rows.append(f"| {NM[imp]} | " + " | ".join(cells) + " |")
    w(f"S_full_{ds}", "\n".join(rows))

rows = [
    "| Subset | Mechanism | Rate | τ_K(imp. RMSE, point regret) | 95% CI "
    "| p (τ_K = 0) | τ_K(imp. RMSE, PHM08) | τ_K(imp. RMSE, risk-aware "
    "regret) |",
    "|---" * 8 + "|",
]
for _, r in dv.sort_values(["dataset", "mech", "rate"]).iterrows():
    rows.append(
        f"| {r.dataset} | {r.mech} | {int(r.rate * 100)}% | "
        f"{r.tau_regret:.2f} | [{r.ci_lo:.2f}, {r.ci_hi:.2f}] | "
        f"{r.p:.2f} | {r.tau_phm08:.2f} | {r.tau_risk:.2f} |"
    )
w("S_div36", "\n".join(rows))

rr = df.groupby(["dataset", "mech", "imputer"]).regret_risk.mean().unstack()
rows = [
    "| Subset | Mech. | " + " | ".join(NM[i] for i in ORDER) + " |",
    "|---" * (len(ORDER) + 2) + "|",
]
for (ds, mech), r in rr.iterrows():
    rows.append(f"| {ds} | {mech} | " + " | ".join(f"{r[i]:.1f}" for i in ORDER) + " |")
w("S_mech", "\n".join(rows))

mi = (
    df[df.imputer.isin(["csdi", "mice"])]
    .groupby(["dataset", "mech", "imputer"])[
        ["coverage_noB", "coverage", "regret_risk_noB", "regret_risk"]
    ]
    .mean()
)
rows = [
    "| Subset | Mechanism | Method | Coverage w/o B | Coverage with B | "
    "Regret w/o B | Regret with B |",
    "|---" * 7 + "|",
]
for (ds, mech, imp), r in mi.iterrows():
    rows.append(
        f"| {ds} | {mech} | {NM[imp].rstrip('†')} | "
        f"{r.coverage_noB:.3f} | {r.coverage:.3f} | "
        f"{r.regret_risk_noB:.1f} | {r.regret_risk:.1f} |"
    )
w("S_abl_mech", "\n".join(rows))

rows = [
    "| Subset | Mechanism | Rate | Chosen by RMSE (seeds 0–4) | Cost on "
    "seeds 5–9 | Chosen by decision cost | Cost on seeds 5–9 |",
    "|---" * 7 + "|",
]
for ds in DS:
    d = df[(df.dataset == ds) & df.imputer.isin(REAL)]
    for (mech, rate), gg in d.groupby(["mech", "rate"]):
        s = gg[gg.seed < 5].groupby("imputer")[["imp_rmse", "regret_risk"]].mean()
        e = gg[gg.seed >= 5].groupby("imputer").regret_risk.mean()
        a, b = s.imp_rmse.idxmin(), s.regret_risk.idxmin()
        rows.append(
            f"| {ds} | {mech} | {int(rate * 100)}% | "
            f"{NM[a].rstrip('†')} | {e[a]:.1f} | "
            f"{NM[b].rstrip('†')} | {e[b]:.1f} |"
        )
w("S_sel_cells", "\n".join(rows))

pd.Series(stats).to_csv(OUT / "stats.csv")
print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in stats.items()})
