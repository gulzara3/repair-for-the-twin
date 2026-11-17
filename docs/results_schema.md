# Results schema

`results/all_results.csv` has one row per configuration. Metrics are computed on the 100, 259, 100 and 248 test engines of FD001 to FD004.

| Column | Meaning |
|---|---|
| `dataset` | C-MAPSS subset, `FD001` to `FD004` |
| `mech` | missingness mechanism: `MCAR`, `MAR`, `MNAR` |
| `rate` | nominal missing rate: 0.1, 0.3, 0.5 |
| `seed` | corruption seed, 0 to 9 |
| `imputer` | `no_repair`, `mean`, `ffill`, `knn`, `mice`, `usgan`, `saits`, `csdi`, `oracle_clean` |
| `imp_rmse`, `imp_mae` | reconstruction error on the injected positions (empty for the two references) |
| `rul_rmse` | RMSE of predicted RUL, in cycles |
| `phm08` | PHM08 challenge score (late predictions penalised more) |
| `coverage`, `width` | coverage and mean width of the conformal interval $\mu \pm \hat q\sqrt{T}$ |
| `regret` | decision regret of the point rule $\mu \le 20$ |
| `regret_risk` | decision regret of the risk-aware rule $\mu - \hat q\sqrt{T} \le 20$ |
| `coverage_noB`, `width_noB`, `regret_risk_noB` | the same with $T = \bar W$ (B term removed); MICE and CSDI only |

Regret is in cost units, with planned maintenance 1, false alarm 2 and missed failure 10.
