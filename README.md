# Financial Inclusion in the Philippines: Global Findex 2025 Microdata

**Exploratory analysis and machine learning on formal financial account ownership among Filipino adults.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Notebook](https://img.shields.io/badge/Jupyter-Colab%20ready-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f7931e)
![statsmodels](https://img.shields.io/badge/statsmodels-inference-lightgrey)

---

## Overview

This project analyzes individual-level survey data from the **World Bank Global Findex 2025** (Philippines, survey year 2024) to understand **who has a formal financial account and who is left behind**.

It is split into two notebooks:

| Notebook | Purpose |
|---|---|
| [`Findex_Philippines_EDA.ipynb`](Findex_Philippines_EDA.ipynb) | Data audit, cleaning, survey-weighted descriptive statistics, and inclusion gaps by group |
| [`Findex_PH_Account_Ownership_ML.ipynb`](Findex_PH_Account_Ownership_ML.ipynb) | Inference and prediction: logistic regression vs. Random Forest, feature importance, and gap analysis |

### Research question

> Which socioeconomic characteristics are associated with owning a **formal** financial account in the Philippines, and where does the digital financial inclusion gap sit?

---

## Data

- **Source:** World Bank, *Global Findex Database 2025* (individual-level microdata) and its *Microdata Codebook*
- **File used:** `Findex_Microdata_2025_updatePhilippines.csv`
- **Unit:** individual adults (15+), survey year 2024
- **Sample:** n = 1,000 respondents, with an effective sample size of about 430 after accounting for survey-weight variation

> **The dataset is not included in this repository.** Download it from the [World Bank Microdata Library](https://microdata.worldbank.org/) / [Global Findex site](https://www.worldbank.org/en/publication/globalfindex) and place the CSV next to the notebooks (see [How to run](#how-to-run)). Please follow the World Bank's terms of use.

---

## Notebook 1: Exploratory Data Analysis

`Findex_Philippines_EDA.ipynb` walks through the full data-understanding workflow:

1. **Structure and integrity:** column blocks by codebook section, unique IDs, duplicate checks
2. **Missingness:** separates *structural* missingness (skip logic: questions asked only of a subgroup) from variables that are entirely empty for the Philippines
3. **Recoding:** converts "don't know" / "refused" codes to missing using the per-variable rules in the codebook
4. **Analysis-ready variables:** labelled demographics and 0/1 inclusion indicators
5. **Survey design:** weight diagnostics, Kish effective sample size, weighted vs. unweighted sample composition
6. **Headline indicators:** weighted estimates with 95% confidence intervals (any account, mobile money, digital payments, phone, internet, foundational ID, and more)
7. **Gaps by group:** gender, age, education, income quintile, urban/rural, workforce status
8. **Behaviour:** saving, borrowing, and how wages, transfers, utilities, and remittances are paid
9. **Barriers:** reasons for having no account, no phone, no smartphone
10. **Financial resilience:** emergency-fund sources, difficulty raising money, greatest financial worries
11. **Connectivity and digital behaviour:** online activities, scams, and harassment
12. **Codebook consistency checks** and a correlation overview
13. **Auto-generated summary** of the headline results and data flags

---

## Notebook 2: Predicting Formal Account Ownership

`Findex_PH_Account_Ownership_ML.ipynb` builds a leakage-aware modelling pipeline.

### Target

`account_fin` = 1 if the adult has an account at a bank or other formal financial institution (credit union, MFI, cooperative, post office) or a debit card. **Mobile-money-only users are coded 0**, because that is where the Philippine "digital" gap sits. A robustness check repeats the analysis for `account`, which includes mobile money.

### Leakage control

Variables that are mechanically tied to account ownership are excluded, for example `dig_account`, `anydigpayment`, debit card ownership, "paid into an account" categories, and any question asked **only** of account owners (their missingness would reveal the target).

### Feature sets (cumulative)

| Set | Contents |
|---|---|
| **A: Socioeconomic** | age, sex, education, income quintile, workforce status, urban/rural, foundational ID |
| **B: + Access** | mobile phone, smartphone, internet use |
| **C: + Digital behaviour** | online activity breadth, digital merchant payment (partly circular, read as association) |

### Methods

- **Logistic regression for inference:** odds ratios, average marginal effects, likelihood-ratio tests, VIF, and a survey-weighted sensitivity check
- **Model comparison:** baseline vs. regularised logistic regression (spline on age) vs. Random Forest vs. gradient boosting, using **repeated stratified 5-fold cross-validation with nested hyper-parameter tuning**
- **Metrics:** ROC-AUC, PR-AUC, Brier score, and weighted AUC
- **Significance testing:** Nadeau & Bengio corrected resampled t-test on paired fold scores
- **Feature importance by five methods,** combined into a consensus ranking: logit LR-test, permutation importance (LR and RF), drop-variable ablation (RF), and RF impurity importance
- **Partial dependence plots** for the shape of key relationships
- **Gap analysis:** out-of-fold propensity quintiles to profile who is least likely to hold an account and which unbanked adults look closest to being included
- **Robustness checks:** alternative feature sets, no foundational ID, weighted training, and the alternative target

---

## Results
 
Weighted estimates for Filipino adults (n = 1,000; effective n ≈ 429). Full details are in the accompanying data analysis report.
 
| Item | Result |
|---|---|
| Adults with a formal (FI) account (weighted) | **33.5%** (95% CI 29.1-38.0) |
| Adults with any account (formal or mobile money) | **50.2%** (mobile money: 28.8%) |
| Gender gap in account ownership | Women ahead of men: 57% vs 43% (any account), 39% vs 28% (formal) |
| Income Q1 vs. Q5 account ownership | Any account 33% vs 64%; formal account 25% vs 46% |
| Logistic regression CV AUC (set B) | **0.672** (± 0.021) |
| Random Forest CV AUC (set B) | **0.672** (± 0.030); no significant difference from logistic regression (p = 0.97) |
| Top 5 characteristics (consensus rank) | 1. Age, 2. Education, 3. In workforce, 4. Female, 5. Income quintile |
---

## How to run

### Option A: Google Colab (easiest)

1. Open a notebook in [Google Colab](https://colab.research.google.com/) (File → Upload notebook, or open it from GitHub).
2. Run the setup cell. When prompted, upload `Findex_Microdata_2025_updatePhilippines.csv`.
3. Run the remaining cells in order.

### Option B: Local

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels joblib jupyter jinja2

# put the CSV next to the notebooks, or set DATA_PATH in the setup cell
jupyter notebook
```

Each notebook looks for a file matching `*Findex*Philippines*.csv` in the working directory. You can also set `DATA_PATH` manually in the setup cell.

**Quick test run:** in the ML notebook, set `FAST = True` to use fewer CV repeats and a smaller hyper-parameter search (faster, but noisier numbers).

### Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
- `scikit-learn` (1.2 or newer), `statsmodels`, `joblib`
- `jinja2` (for styled tables in `pandas`)

---

## Repository structure

```
.
├── Findex_Philippines_EDA.ipynb
├── Findex_PH_Account_Ownership_ML.ipynb
├── README.md
└── data/                      # not tracked; place the Findex CSV here or next to the notebooks
```

---

## Data notes and gotchas

These came up while working through the codebook:

- `female` is coded **1 = female, 2 = male** (not 0/1)
- The codebook's `internet` indicator is named `internet_use` in this file
- "Don't know" / "refused" codes differ by variable (3/4, 4/5, 5/6, 7/8, 8/9, 98/99) and were mapped per the codebook
- Some variables are entirely empty for the Philippines and some subgroup questions have very small bases (treat estimates with n < ~50 as indicative)
- `fin17b` and `fin22a_1` are answered by more respondents than own mobile money, and `con24` contains an undocumented code; see the consistency checks in the EDA notebook
- `fin11*` (reasons for no account) refers to financial-institution accounts and excludes mobile money

---

## Limitations

- **Association, not causation.** Income, education, and ID are likely intertwined with unobserved factors (branch proximity, trust, informal-sector work).
- **Small sample with uneven weights.** n = 1,000 with an effective n of about 430, so differences smaller than roughly 0.02 AUC are within cross-validation noise.
- **No strata or PSU identifiers.** Confidence intervals use the Kish effective sample size and ignore clustering, so they are probably too narrow.
- **Leakage.** Digital-payment and account-use variables are partly consequences of owning an account, so feature set C should be read as description, not prediction.
- **Collinearity** among phone, smartphone, and internet use makes their individual importance unstable even when their joint importance is clear.
- **Self-reported, cross-sectional** survey data.

---

## Acknowledgements

- World Bank, *Global Findex Database 2025* and its microdata codebook
- Statistical references: Kish (1965) effective sample size; Nadeau & Bengio (2003) corrected resampled t-test

## License

Code is released under the [MIT License](LICENSE) (add a `LICENSE` file). The Findex data is **not** covered by this license and remains subject to the World Bank's terms of use.
