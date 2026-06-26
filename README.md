# Critical Reproduction of a Credit-Card Fraud Detector

Final project for **Data Science in Cyber** (Dr. Uri Itai). This repository critically reproduces and
stress-tests a published machine-learning credit-card fraud detector, testing whether the author's
claim of ~100% accuracy is supported by the evidence.

**Author:** Yarin Saraga (330744392)

## Summary

The evaluated source trains KNN / Decision-Tree models on the Kaggle/ULB credit-card dataset and reports
~100% accuracy. We show that this number is an artifact of extreme class imbalance (fraud is 0.17% of
rows, so a do-nothing model already scores 99.83%), and that judged by appropriate metrics the source's
model is the weakest we tested. A properly imbalance-aware XGBoost reaches a genuine **PR-AUC of ~0.87**.

Three experiments form the spine of the evaluation:
1. **The accuracy illusion** — accuracy cannot distinguish the models; PR-AUC can.
2. **Leakage and duplicate rows** — real malpractice, but measured to be numerically negligible here.
3. **k=1 memorization and imbalance handling** — k=5 and an imbalance-aware model do far better by PR-AUC.

## Links

- **Source under evaluation:** https://github.com/shakiliitju/Credit-Card-Fraud-Detection-Using-Machine-Learning
- **Dataset (original):** Kaggle — *Credit Card Fraud Detection* (ULB, Dal Pozzolo et al.):
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Repository structure

```
fraud_critical_repro.ipynb            # full analysis notebook (run top to bottom)
Fraud_Critical_Evaluation_Report.pdf  # written report (8 sections) — final deliverable
Fraud_Critical_Evaluation_Report.docx # editable report source
DataFiles/creditcard.csv           # dataset, gzipped (pandas reads it directly)
requirements.txt                      # dependencies
```

## How to run

**Google Colab (recommended).** Open `fraud_critical_repro.ipynb`, set `REPO_URL` in the first code cell
to this repository's URL, then **Runtime → Restart and run all**. The first cell clones the repo so the
relative `DataFiles/` path resolves. A full run takes ~3–4 minutes (KNN and Random Forest on 285k rows).

**Local.**
```bash
pip install -r requirements.txt
jupyter notebook fraud_critical_repro.ipynb   # run from the repo root so DataFiles/ resolves
```

## Dataset source

The dataset is the public Kaggle/ULB credit-card fraud collection (284,807 transactions, 492 frauds,
28 anonymised PCA features plus `Time` and `Amount`). It is committed here gzipped (`DataFiles/creditcard.csv`,
~43 MB) so the notebook is self-contained; `pandas.read_csv` reads the `.gz` directly.
