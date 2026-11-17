# Data

The NASA C-MAPSS turbofan data are not stored in this repository. Download them with

```bash
repair4twin download
```

This places `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` (x = 1 to 4) in `data/cmapss/`. You can also copy the twelve files there by hand. The dataset is published by the NASA Prognostics Center of Excellence ([Saxena et al., 2008](https://doi.org/10.1109/PHM.2008.4711414)).
