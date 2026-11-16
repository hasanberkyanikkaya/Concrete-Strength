# Concrete Compressive Strength Prediction with ANNs

Predicting concrete compressive strength (MPa) from mix design and curing age
(UCI Concrete Compressive Strength dataset, I-Cheng Yeh, 1998).

## Structure
```
data/raw/Concrete_Data.xls   UCI dataset (1,030 rows, 8 inputs, 1 output)
src/config.py                paths, columns, split, seeds, hyperparameter grids
src/data.py                  cleaning, mix ID, features, grouped split, scaling
src/models.py                LR, RF, SVR, MLP-A/B/C (PyTorch), validation tuning
src/evaluate.py              RMSE / MAE / R², age & strength breakdowns
src/plotting.py              all figures
run_experiment.py            end-to-end pipeline
results/                     tables, summary.md, figures/ (generated)
```

## Method
- Exact duplicates removed; IQR outliers reported but kept (physically plausible).
- `mix_id` = identical 7-component proportions. 70/15/15 split **grouped by mix**,
  so the same mix never appears at different ages in different subsets.
- Features: 7 components + `log(age)` + water/cement + water/binder;
  StandardScaler fitted on training data only.
- Models: Linear Regression (reference), Random Forest, SVR (RBF),
  MLP-A (32), MLP-B (128-64-32), MLP-C (MLP-B + dropout + L2).
  ANNs: ReLU, linear output, MSE, Adam, early stopping on validation loss.
- Hyperparameters chosen on validation RMSE; **test set evaluated once**.
- Robustness: the whole protocol (grouped split → tuning on that split's validation
  set → test) is repeated on 20 seeds, so no split's test mixes influence its model
  choice; mean ± std reported.
- Model comparison: paired split-wise RMSE differences with the corrected resampled
  t-test (Nadeau & Bengio, 2003) and Wilcoxon signed-rank test, Holm-adjusted.
- Error analysis by curing age (1-3, 7, 14-28, 56-100, 120-365 d) and actual strength
  range (RMSE, MAE, bias, MAPE), on the main test set and pooled over all 20 splits.

## Run
```
python -m pip install --user -r requirements.txt
python run_experiment.py --fast   # smoke test
python run_experiment.py          # full run
```
Tested with Python 3.15 + PyTorch (CPU).

## Outputs (in `results/`)
`test_results.csv`, `best_params.json`, `tuning_<model>.csv`, `errors_by_age_group.csv`,
`errors_by_strength_group.csv`, `worst_test_cases.csv`,
`errors_inside_vs_outside_train_range.csv`, `repeated_splits_summary.csv`,
`repeated_splits_raw.csv`, `repeated_best_params.csv`, `repeated_test_predictions.csv`,
`paired_comparisons.csv`, `repeated_errors_by_age_group.csv`,
`repeated_errors_by_strength_group.csv`, `summary.md`, and `figures/*.png`.

## External resources and references
**Dataset**
- Yeh, I-C. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5PK67 —
  https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
- Yeh, I-C. (1998). Modeling of strength of high performance concrete using artificial
  neural networks. *Cement and Concrete Research*, 28(12), 1797-1808.
  (Required citation for reuse of the dataset.)

**Libraries** (versions in `requirements.txt`)
- PyTorch — Paszke et al. (2019), *PyTorch: An imperative style, high-performance deep
  learning library*, NeurIPS 32.
- scikit-learn — Pedregosa et al. (2011), *Scikit-learn: Machine learning in Python*,
  JMLR 12, 2825-2830.
- NumPy, pandas, Matplotlib, xlrd.

**Methods**
- Adam optimizer — Kingma & Ba (2015), *Adam: A method for stochastic optimization*, ICLR.
- Dropout — Srivastava et al. (2014), JMLR 15, 1929-1958.
- Random Forest — Breiman (2001), *Machine Learning*, 45, 5-32.
- SVR — Drucker et al. (1997), NeurIPS 9; Smola & Schölkopf (2004), *Statistics and
  Computing*, 14, 199-222.
- Corrected resampled t-test — Nadeau & Bengio (2003), *Inference for the generalization
  error*, Machine Learning, 52, 239-281.
- Holm (1979), *A simple sequentially rejective multiple test procedure*, Scandinavian
  Journal of Statistics, 6, 65-70.

**Reused resources / tools**
- No code was copied from external projects.
- AI assistance: code was developed with the help of an AI coding assistant
  (Claude Code, Anthropic). <!-- Check the course policy on AI use and edit as needed. -->

## Citation
I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial
neural networks," Cement and Concrete Research, 28(12), 1797-1808 (1998).
