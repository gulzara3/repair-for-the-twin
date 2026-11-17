"""Command-line entry point: ``repair4twin {download,run,summary}``."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from repair4twin.config import SUBSETS, Config


def _download(args) -> None:
    from repair4twin.data import fetch_cmapss

    fetch_cmapss(Path(args.data_dir))


def _run(args) -> None:
    import torch

    from repair4twin.data import fetch_cmapss
    from repair4twin.harness import run_dataset
    from repair4twin.store import ResultStore
    from repair4twin.twin import get_device, set_global_determinism

    cfg = Config.quick() if args.quick else Config()
    over = {"data_dir": Path(args.data_dir)}
    if args.subsets:
        over["subsets"] = tuple(args.subsets)
    if args.out_dir:
        over["out_dir"] = Path(args.out_dir)
    cfg = replace(cfg, **over)

    set_global_determinism(cfg.seed)
    device = get_device()
    print(f"device: {device} | subsets: {', '.join(cfg.subsets)} | output: {cfg.out_dir}")
    if device == "cpu":
        print("[warn] no GPU found; the full run is only practical on a GPU.")
    fetch_cmapss(cfg.data_dir)
    store = ResultStore(cfg.out_dir)
    generator = torch.Generator().manual_seed(cfg.seed)
    q = {}
    for name in cfg.subsets:
        _, q[name] = run_dataset(name, cfg, store, device, generator)
    print(f"conformal multipliers: {json.dumps(q)}")
    print(f"results: {store.results_csv}")


def _summary(args) -> None:
    import pandas as pd

    from repair4twin.stats import summary

    s = summary(pd.read_csv(args.results))
    d, b = s["divergence"], s["b_term"]
    print(f"rows: {s['rows']}")
    print(
        f"mean Kendall tau with RUL RMSE {d['mean_tau_rul_rmse']:.2f}, PHM08 "
        f"{d['mean_tau_phm08']:.2f}, point regret {d['mean_tau_regret']:.2f}"
    )
    print(
        f"accuracy layer agrees more in {d['cells_accuracy_above_decision']}/{d['n_cells']} "
        f"cells (one-sided Wilcoxon p = {d['p_phm08']:.1e})"
    )
    lo, hi = s["policy_ratio_iqr"]
    print(
        f"point / risk-aware regret: median {s['policy_ratio_median']:.2f} (IQR {lo:.2f}-{hi:.2f})"
    )
    print(
        f"B term vs coverage without B: Spearman {b['spearman_subset_means']:.2f} over "
        f"{b['n_subset_means']} subset means, {b['spearman_cells']:.2f} over "
        f"{b['n_cells']} cells"
    )
    for r in s["selection"]:
        print(
            f"{r['dataset']}: out-of-sample saving {r['saving_pct']:.1f}% "
            f"(RMSE pick {r['pick_by_rmse']}, decision pick {r['pick_by_decision']}), "
            f"per-cell {r['per_cell_saving_pct']:+.1f}%"
        )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="repair4twin", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download NASA C-MAPSS")
    d.add_argument("--data-dir", default="data/cmapss")
    d.set_defaults(func=_download)

    r = sub.add_parser("run", help="run the experiment (GPU recommended)")
    r.add_argument("--quick", action="store_true", help="smoke test on FD001 (~25 min)")
    r.add_argument("--subsets", nargs="+", choices=SUBSETS)
    r.add_argument("--data-dir", default="data/cmapss")
    r.add_argument("--out-dir", default=None, help="default: runs/default or runs/quick")
    r.set_defaults(func=_run)

    s = sub.add_parser("summary", help="print the paper's headline numbers")
    s.add_argument("results", nargs="?", default="results/all_results.csv")
    s.set_defaults(func=_summary)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
