# Experiment summary

## Data

```json
{
  "n_raw": 1030,
  "missing_values": 0,
  "exact_duplicates_removed": 25,
  "n_after_cleaning": 1005,
  "iqr_outliers_kept": {
    "cement": 0,
    "slag": 2,
    "fly_ash": 0,
    "water": 15,
    "superplasticizer": 10,
    "coarse_agg": 0,
    "fine_agg": 5,
    "age": 59,
    "strength": 8
  },
  "n_unique_mixes": 428,
  "mixes_tested_at_multiple_ages": 182,
  "rows_in_multi_age_mixes": 759,
  "replicate_rows_same_mix_and_age": 9
}
```

## Mix-grouped split

| subset | rows | row_share | mixes | strength_mean | strength_std | n_1-3 d | n_7 d | n_14-28 d | n_56-100 d | n_120-365 d |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 705 | 0.701 | 298 | 35.107 | 15.800 | 92 | 86 | 336 | 145 | 46 |
| val | 162 | 0.161 | 65 | 35.163 | 16.783 | 22 | 17 | 76 | 37 | 10 |
| test | 138 | 0.137 | 65 | 36.086 | 18.133 | 17 | 19 | 69 | 27 | 6 |

## Test-set results (single evaluation)

| model | RMSE | MAE | R2 | train_RMSE | val_RMSE | size | size_unit | train_time_s |
|---|---|---|---|---|---|---|---|---|
| LinearRegression | 8.362 | 5.934 | 0.786 | 6.807 | 6.781 | 11 | coefficients | 0.001 |
| RandomForest | 6.523 | 4.472 | 0.870 | 1.776 | 5.442 | 263914 | tree nodes | 0.202 |
| SVR | 6.702 | 4.587 | 0.862 | 3.904 | 5.435 | 417 | support vectors | 0.018 |
| MLP-A | 7.554 | 5.308 | 0.825 | 4.505 | 4.843 | 385 | weights | 4.839 |
| MLP-B | 6.683 | 4.546 | 0.863 | 3.233 | 4.604 | 11777 | weights | 3.678 |
| MLP-C | 6.893 | 4.682 | 0.854 | 3.621 | 4.514 | 11777 | weights | 5.351 |

## Best hyperparameters (chosen on validation RMSE)

- **LinearRegression**: `{}`
- **RandomForest**: `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 0.5}`
- **SVR**: `{'C': 100, 'gamma': 0.05, 'epsilon': 2.0}`
- **MLP-A**: `{'lr': 0.003, 'batch_size': 16}`
- **MLP-B**: `{'lr': 0.001, 'batch_size': 32}`
- **MLP-C**: `{'lr': 0.001, 'batch_size': 32, 'dropout': 0.1, 'l2': 0.001}`

## Test RMSE by curing-age group (MPa)

| model | 1-3 d | 7 d | 14-28 d | 56-100 d | 120-365 d |
|---|---|---|---|---|---|
| LinearRegression | 3.461 | 7.934 | 9.469 | 7.224 | 10.444 |
| RandomForest | 3.663 | 5.099 | 7.690 | 6.234 | 1.820 |
| SVR | 3.780 | 4.894 | 8.202 | 5.558 | 2.046 |
| MLP-A | 4.423 | 4.914 | 8.387 | 8.535 | 6.612 |
| MLP-B | 5.087 | 4.655 | 7.510 | 7.166 | 2.358 |
| MLP-C | 5.149 | 4.513 | 7.843 | 7.130 | 4.136 |

## Extrapolation: test RMSE inside vs. outside training input range (10 rows outside)

| model | inside | outside |
|---|---|---|
| LinearRegression | 8.049 | 11.645 |
| RandomForest | 5.861 | 12.140 |
| SVR | 6.083 | 12.090 |
| MLP-A | 6.384 | 16.301 |
| MLP-B | 5.483 | 15.216 |
| MLP-C | 5.646 | 15.739 |

## Hardest test samples (largest mean |error| over all models)

| mix_id | age | w_c | actual | mean_abs_err | outside_train_range |
|---|---|---|---|---|---|
| 76 | 28 | 0.460 | 81.751 | 24.511 | coarse_agg |
| 78 | 28 | 0.366 | 78.800 | 23.381 | - |
| 90 | 28 | 0.950 | 49.249 | 19.151 | coarse_agg |
| 79 | 28 | 0.314 | 41.369 | 17.330 | - |
| 77 | 28 | 0.386 | 64.018 | 17.259 | - |
| 0 | 28 | 0.300 | 79.986 | 16.665 | - |
| 30 | 91 | 0.406 | 56.496 | 15.469 | coarse_agg |
| 198 | 7 | 0.626 | 30.957 | 14.882 | - |
| 126 | 28 | 0.396 | 72.099 | 14.418 | - |
| 30 | 56 | 0.406 | 54.896 | 13.944 | coarse_agg |

## Repeated group splits (20 seeds, re-tuned on each split; mean / std)

| model | RMSE_mean | RMSE_std | MAE_mean | MAE_std | R2_mean | R2_std |
|---|---|---|---|---|---|---|
| LinearRegression | 7.393 | 0.597 | 5.699 | 0.436 | 0.786 | 0.032 |
| RandomForest | 5.454 | 0.781 | 3.925 | 0.538 | 0.883 | 0.028 |
| SVR | 5.806 | 0.806 | 4.168 | 0.558 | 0.867 | 0.032 |
| MLP-A | 6.553 | 0.959 | 4.861 | 0.627 | 0.830 | 0.047 |
| MLP-B | 5.937 | 0.867 | 4.236 | 0.553 | 0.860 | 0.038 |
| MLP-C | 5.749 | 0.854 | 4.167 | 0.600 | 0.870 | 0.031 |

## Paired comparisons (RMSE_a - RMSE_b, negative = a better)

Corrected resampled t-test (Nadeau & Bengio, 2003), n_test/n_train = 0.213; CI uses the corrected standard error; `a_better_count` = splits (of 20) where a had lower RMSE; p-values Holm-adjusted.

| comparison | mean_diff | std_diff | ci95_low | ci95_high | a_better_count | p_corrected_t | p_corrected_t_holm | p_wilcoxon_holm |
|---|---|---|---|---|---|---|---|---|
| MLP-B - MLP-A | -0.616 | 0.499 | -1.152 | -0.081 | 19 | 0.026 | 0.158 | 0.000 |
| MLP-C - MLP-B | -0.188 | 0.587 | -0.817 | 0.442 | 12 | 0.540 | 1.000 | 0.246 |
| MLP-B - RandomForest | 0.483 | 0.667 | -0.232 | 1.199 | 4 | 0.174 | 0.695 | 0.019 |
| MLP-B - SVR | 0.131 | 0.512 | -0.418 | 0.681 | 9 | 0.623 | 1.000 | 0.277 |
| RandomForest - SVR | -0.352 | 0.563 | -0.956 | 0.253 | 16 | 0.238 | 0.714 | 0.041 |
| MLP-A - LinearRegression | -0.840 | 0.741 | -1.634 | -0.045 | 17 | 0.039 | 0.197 | 0.001 |

## Pooled test error by curing-age group (20 splits)

Predictions pooled over all splits; `n` = pooled predictions, `n_unique` = distinct samples.

| group | n | n_unique |
|---|---|---|
| 1-3 d | 379 | 124 |
| 7 d | 362 | 116 |
| 14-28 d | 1446 | 463 |
| 56-100 d | 619 | 198 |
| 120-365 d | 191 | 60 |

**RMSE**

| model | 1-3 d | 7 d | 14-28 d | 56-100 d | 120-365 d |
|---|---|---|---|---|---|
| LinearRegression | 5.577 | 7.540 | 7.411 | 7.125 | 10.674 |
| RandomForest | 4.441 | 5.187 | 5.702 | 6.161 | 3.975 |
| SVR | 5.094 | 5.272 | 6.313 | 5.878 | 4.334 |
| MLP-A | 5.285 | 5.466 | 6.741 | 7.030 | 8.477 |
| MLP-B | 5.031 | 4.988 | 6.263 | 6.477 | 5.877 |
| MLP-C | 4.590 | 4.663 | 6.110 | 6.293 | 6.036 |

**MAPE**

| model | 1-3 d | 7 d | 14-28 d | 56-100 d | 120-365 d |
|---|---|---|---|---|---|
| LinearRegression | 35.388 | 33.078 | 17.283 | 11.512 | 25.748 |
| RandomForest | 23.915 | 17.902 | 11.642 | 9.960 | 7.475 |
| SVR | 25.975 | 17.474 | 12.937 | 9.291 | 7.538 |
| MLP-A | 28.288 | 21.173 | 15.086 | 10.960 | 17.843 |
| MLP-B | 24.617 | 16.626 | 12.805 | 10.093 | 10.083 |
| MLP-C | 22.700 | 16.192 | 12.796 | 9.860 | 10.307 |

**bias**

| model | 1-3 d | 7 d | 14-28 d | 56-100 d | 120-365 d |
|---|---|---|---|---|---|
| LinearRegression | 0.007 | 1.802 | -0.534 | -2.088 | 6.012 |
| RandomForest | 0.916 | 0.256 | -0.163 | -0.981 | -0.533 |
| SVR | 0.264 | 0.535 | -0.195 | -0.547 | 0.111 |
| MLP-A | 0.735 | 1.332 | -0.237 | -0.305 | 3.525 |
| MLP-B | -0.032 | 0.148 | -0.057 | -0.193 | 0.701 |
| MLP-C | 0.043 | 0.403 | -0.285 | -0.231 | 1.185 |

## Pooled test error by actual-strength group (20 splits)

Predictions pooled over all splits; `n` = pooled predictions, `n_unique` = distinct samples.

| group | n | n_unique |
|---|---|---|
| <20 MPa | 595 | 190 |
| 20-40 MPa | 1358 | 434 |
| 40-60 MPa | 822 | 266 |
| >60 MPa | 222 | 71 |

**RMSE**

| model | <20 MPa | 20-40 MPa | 40-60 MPa | >60 MPa |
|---|---|---|---|---|
| LinearRegression | 6.266 | 6.157 | 7.988 | 13.027 |
| RandomForest | 3.870 | 4.512 | 5.750 | 11.032 |
| SVR | 4.047 | 5.015 | 6.552 | 10.248 |
| MLP-A | 4.935 | 5.661 | 7.887 | 10.018 |
| MLP-B | 4.114 | 5.217 | 7.113 | 9.316 |
| MLP-C | 3.702 | 4.963 | 6.948 | 9.496 |

**MAPE**

| model | <20 MPa | 20-40 MPa | 40-60 MPa | >60 MPa |
|---|---|---|---|---|
| LinearRegression | 44.167 | 15.789 | 13.714 | 15.441 |
| RandomForest | 24.754 | 10.891 | 9.328 | 12.548 |
| SVR | 24.824 | 11.980 | 10.235 | 11.781 |
| MLP-A | 31.421 | 13.552 | 13.133 | 11.251 |
| MLP-B | 23.333 | 12.315 | 11.052 | 10.570 |
| MLP-C | 22.353 | 11.900 | 11.262 | 10.462 |

**bias**

| model | <20 MPa | 20-40 MPa | 40-60 MPa | >60 MPa |
|---|---|---|---|---|
| LinearRegression | 1.860 | 2.176 | -2.660 | -9.626 |
| RandomForest | 1.849 | 1.101 | -1.609 | -8.005 |
| SVR | 1.157 | 0.961 | -1.139 | -6.133 |
| MLP-A | 1.996 | 1.562 | -1.209 | -6.359 |
| MLP-B | 0.930 | 1.030 | -1.008 | -5.185 |
| MLP-C | 1.189 | 0.888 | -1.039 | -5.524 |

## Hyperparameter stability (most frequent choice across splits)

| model | params | splits |
|---|---|---|
| LinearRegression | {} | 20 |
| RandomForest | {"n_estimators": 300, "max_depth": null, "min_samples_leaf": 1, "max_features": 0.5} | 6 |
| SVR | {"C": 300, "gamma": 0.05, "epsilon": 2.0} | 6 |
| MLP-A | {"lr": 0.003, "batch_size": 16} | 8 |
| MLP-B | {"lr": 0.003, "batch_size": 16} | 8 |
| MLP-C | {"lr": 0.003, "batch_size": 16, "dropout": 0.1, "l2": 0.001} | 4 |
