"""Regenerate every figure that is computed from the results table.

Usage (from the repository root):

    python scripts/make_figures.py

Writes Figs. 2, 3, 4, 7 and 8 to figures/paper/ and Figs. S1-S4 to
figures/supplementary/. Fig. 1 and the graphical abstract are drawn from the
HTML sources in figures/source/ (see scripts/render_diagrams.py). Figs. 5, 6
and 9 need the per-engine prediction archives written by a full run, and are
produced by Section 7 of the notebook.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--results", default="results/all_results.csv")
ap.add_argument("--divergence", default="results/divergence_all_datasets.csv")
ap.add_argument("--out", default="figures")
args = ap.parse_args()
FIG, SUP = Path(args.out) / "paper", Path(args.out) / "supplementary"
FIG.mkdir(parents=True, exist_ok=True)
SUP.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.results)
dv = pd.read_csv(args.divergence)
REAL = ["mean", "ffill", "knn", "mice", "usgan", "saits", "csdi"]
DS = ["FD001", "FD002", "FD003", "FD004"]
NM = {
    "mean": "mean",
    "ffill": "forward-fill",
    "knn": "KNN",
    "mice": "MICE",
    "usgan": "US-GAN",
    "saits": "SAITS",
    "csdi": "CSDI",
    "oracle_clean": "clean input",
    "no_repair": "no repair",
}
CB = ["#0173b2", "#de8f05", "#029e73", "#d55e00"]
ACC = {"knn": "#d55e00", "usgan": "#0173b2"}
GREY, LGREY = "#555555", "#9a9a9a"
TK = r"$\tau_K$"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.5,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
def fig2():
    a = df[df.imputer.isin(REAL)].groupby(["dataset", "imputer"])[["imp_rmse", "regret"]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for ax, ds in zip(axes.flat, DS, strict=True):
        s = a.loc[ds]
        up, dn = (5, 4), (7, -12)
        hi = s.loc["saits", "regret"] > s.loc["csdi", "regret"]
        offs = {"saits": up if hi else dn, "csdi": dn if hi else up}
        if ds == "FD003":
            offs["ffill"] = (-8, -13)
        for imp, r in s.iterrows():
            ax.scatter(
                r.imp_rmse,
                r.regret,
                s=55,
                color=ACC.get(imp, LGREY),
                edgecolor="white",
                lw=1,
                zorder=3,
                marker="D" if imp in ACC else "o",
            )
            ax.annotate(
                NM[imp],
                (r.imp_rmse, r.regret),
                fontsize=7.5,
                xytext=offs.get(imp, (5, 3)),
                textcoords="offset points",
                color=GREY,
            )
        t = kendalltau(s.imp_rmse, s.regret)[0]
        ax.set_title(f"{ds}   ({TK} over subset means = {t:.2f})", fontsize=9)
        ax.set_xlabel("Imputation RMSE (normalised units)", fontsize=8.5)
        ax.set_ylabel("Point-policy decision regret", fontsize=8.5)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.18)
    fig.tight_layout()
    save(fig, FIG / "Fig2_repair_vs_decision.png")


# ------------------------------------------------------------------
# Fig. 3  two-layer tau
# ------------------------------------------------------------------
def fig3():
    rows = []
    for ds in DS:
        sub = df[(df.dataset == ds) & df.imputer.isin(REAL)]
        for _, g in sub.groupby(["mech", "rate"]):
            a = g.groupby("imputer")[
                ["imp_rmse", "rul_rmse", "phm08", "regret", "regret_risk"]
            ].mean()
            rows.append(
                dict(
                    ds=ds,
                    rul=kendalltau(a.imp_rmse, a.rul_rmse)[0],
                    phm=kendalltau(a.imp_rmse, a.phm08)[0],
                    reg=kendalltau(a.imp_rmse, a.regret)[0],
                    risk=kendalltau(a.imp_rmse, a.regret_risk)[0],
                )
            )
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x, w = np.arange(4), 0.2
    series = [
        ("rul", "RUL RMSE", "//"),
        ("phm", "PHM08", "\\\\"),
        ("reg", "point-policy regret", ""),
        ("risk", "risk-aware regret", ".."),
    ]
    for k, (c, lab, h) in enumerate(series):
        g = t.groupby("ds")[c]
        m = g.mean().reindex(DS).values
        ax.bar(
            x + (k - 1.5) * w,
            m,
            w * 0.92,
            yerr=[m - g.min().reindex(DS).values, g.max().reindex(DS).values - m],
            capsize=2.5,
            color=CB[k],
            hatch=h,
            edgecolor="white",
            lw=0.6,
            label=f"{TK}(imputation RMSE, {lab})",
            error_kw=dict(lw=0.8, ecolor="#333333"),
        )
    ax.axhline(1, ls=":", color=GREY, lw=0.8)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(DS)
    ax.set_ylim(-0.45, 1.08)
    ax.set_ylabel(f"Mean {TK} over 9 cells\n(bars: min–max across cells)")
    ax.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    save(fig, FIG / "Fig3_two_layer_tau.png")


# ------------------------------------------------------------------
# Fig. 4 + S1  tau heatmaps
# ------------------------------------------------------------------
def heat(ax, ds, cbar, fs):
    d = dv[dv.dataset == ds]
    piv = d.pivot(index="mech", columns="rate", values="tau_regret")
    lab = np.empty(piv.shape, dtype=object)
    for i, m in enumerate(piv.index):
        for j, r in enumerate(piv.columns):
            q = d[(d.mech == m) & (d.rate == r)].iloc[0]
            lab[i, j] = f"{q.tau_regret:.2f}\n[{q.ci_lo:.2f}, {q.ci_hi:.2f}]"
    sns.heatmap(
        piv,
        annot=lab,
        fmt="",
        cmap="Blues",
        vmin=-0.2,
        vmax=1,
        linewidths=2,
        linecolor="white",
        cbar=cbar,
        ax=ax,
        annot_kws={"fontsize": fs},
        cbar_kws={"label": f"Kendall {TK} (1 = rankings agree)"} if cbar else None,
    )
    ax.set_xticklabels([f"{int(c * 100)}%" for c in piv.columns])
    ax.set_xlabel("Missingness rate")
    ax.set_ylabel("Mechanism")


def fig4():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    heat(ax, "FD004", True, 8)
    fig.tight_layout()
    save(fig, FIG / "Fig4_tau_heatmap_FD004.png")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    for ax, ds in zip(axes.flat, DS, strict=True):
        heat(ax, ds, False, 7)
        ax.set_title(ds, fontsize=9)
    fig.tight_layout()
    save(fig, SUP / "S1_tau_heatmaps.png")


# ------------------------------------------------------------------
# Fig. 7  point vs risk-aware policy
# ------------------------------------------------------------------
def fig7():
    imps = ["oracle_clean", "saits", "csdi"]
    pol = (
        df[df.imputer.isin(imps)].groupby(["dataset", "imputer"])[["regret", "regret_risk"]].mean()
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.1))
    for ax, ds in zip(axes, DS, strict=True):
        for k, imp in enumerate(imps):
            r = pol.loc[(ds, imp)]
            first = k == 0 and ds == "FD001"
            ax.plot([k, k], [r.regret, r.regret_risk], color="#bbbbbb", lw=2, zorder=1)
            ax.scatter(
                k,
                r.regret,
                s=40,
                color="#7f7f7f",
                zorder=3,
                label="point policy" if first else None,
            )
            ax.scatter(
                k,
                r.regret_risk,
                s=46,
                color=CB[0],
                zorder=3,
                marker="D",
                label="risk-aware policy" if first else None,
            )
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([NM[i] for i in imps], rotation=35, ha="right", fontsize=8)
        ax.set_title(ds, fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean decision regret")
    fig.legend(loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, FIG / "Fig7_policy_effect.png")


# ------------------------------------------------------------------
# Fig. 8  value of B vs coverage without B
# ------------------------------------------------------------------
def fig8():
    mi = df[df.imputer.isin(["csdi", "mice"])].copy()
    mi["dB"] = mi.regret_risk - mi.regret_risk_noB
    cell = (
        mi.groupby(["dataset", "imputer", "mech", "rate"])[["coverage_noB", "dB"]]
        .mean()
        .reset_index()
    )
    agg = mi.groupby(["dataset", "imputer"])[["coverage_noB", "dB"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for k, imp in enumerate(["csdi", "mice"]):
        mk = "o" if imp == "csdi" else "s"
        c, a = cell[cell.imputer == imp], agg[agg.imputer == imp]
        ax.scatter(
            c.coverage_noB,
            c.dB,
            s=14,
            color=CB[k],
            alpha=0.35,
            lw=0,
            marker=mk,
            label=f"{NM[imp]}: one mechanism × rate cell",
        )
        ax.scatter(
            a.coverage_noB,
            a.dB,
            s=75,
            color=CB[k],
            edgecolor="white",
            lw=1.5,
            zorder=3,
            marker=mk,
            label=f"{NM[imp]}: subset mean",
        )
    lbl = {("FD002", "csdi"): (6, -10), ("FD002", "mice"): (-38, 6), ("FD004", "mice"): (6, -12)}
    for (ds, imp), off in lbl.items():
        r = agg[(agg.dataset == ds) & (agg.imputer == imp)].iloc[0]
        ax.annotate(
            f"{NM[imp]} {ds}",
            (r.coverage_noB, r.dB),
            fontsize=7,
            xytext=off,
            textcoords="offset points",
            color="#333333",
        )
    ax.axhline(0, color="#333333", lw=0.8)
    ax.text(
        0.99,
        0.04,
        "below 0: B lowers decision cost",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.5,
        color="#333333",
    )
    ax.set_xlabel("Interval coverage without B (T = W̄)")
    ax.set_ylabel("Risk-aware regret: with B − without B")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, FIG / "Fig8_B_conditional.png")


# ------------------------------------------------------------------
# Supplementary S2-S4
# ------------------------------------------------------------------
def supp():
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.4))
    for ax, ds in zip(axes, DS, strict=True):
        s = (
            df[(df.dataset == ds) & (df.mech == "MNAR") & (df.rate == 0.5) & df.imputer.isin(REAL)]
            .groupby("imputer")[["imp_rmse", "regret"]]
            .mean()
        )
        r1, r2 = s.imp_rmse.rank(), s.regret.rank()
        for imp in s.index:
            ax.plot(
                [0, 1],
                [r1[imp], r2[imp]],
                "-o",
                color=ACC.get(imp, LGREY),
                lw=2.2 if imp in ACC else 1.3,
                ms=4,
            )
            ax.text(-0.06, r1[imp], NM[imp], ha="right", va="center", fontsize=6.8, color=GREY)
            ax.text(1.06, r2[imp], NM[imp], ha="left", va="center", fontsize=6.8, color=GREY)
        ax.invert_yaxis()
        ax.set_xlim(-1.0, 1.95)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["imp. RMSE", "regret"], fontsize=8)
        ax.set_yticks([])
        ax.set_title(ds, fontsize=9)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, SUP / "S2_bump_MNAR50.png")

    cw = df.groupby(["dataset", "imputer"])[["coverage", "width"]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for ax, ds in zip(axes.flat, DS, strict=True):
        for imp, r in cw.loc[ds].iterrows():
            mi_ = imp in ("csdi", "mice")
            ax.scatter(
                r.width,
                r.coverage,
                s=45,
                color=CB[0] if mi_ else LGREY,
                edgecolor="white",
                zorder=3,
                marker="s" if mi_ else "o",
            )
            ax.annotate(
                NM[imp],
                (r.width, r.coverage),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
                color=GREY,
            )
        ax.axhline(0.95, ls="--", color="#777777", lw=0.8)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("mean interval width (cycles)", fontsize=8.5)
        ax.set_ylabel("coverage", fontsize=8.5)
        ax.margins(x=0.15)
    fig.tight_layout()
    save(fig, SUP / "S3_coverage_width.png")

    fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.1))
    for ax, ds in zip(axes, DS, strict=True):
        d = df[df.dataset == ds]
        for imp, c, mk in [("saits", CB[1], "o"), ("csdi", CB[0], "s")]:
            s = d[d.imputer == imp].groupby(["mech", "rate"]).regret_risk.mean()
            ax.plot(range(9), s.values, marker=mk, color=c, lw=1.2, ms=4, label=NM[imp])
        ref = d[d.imputer == "oracle_clean"].regret_risk.mean()
        ax.axhline(ref, color=GREY, ls=":", lw=1, label="clean input")
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"{m}\n{int(r * 100)}" for m, r in s.index], fontsize=6)
        ax.set_title(ds, fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("risk-aware regret")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, SUP / "S4_regret_by_cell.png")


for f in (fig2, fig3, fig4, fig7, fig8, supp):
    f()
print("figures written")
