# FairLens: an AI fairness audit dashboard

A small Streamlit app that audits a hiring classifier for group fairness. Built for UCS421 (Ethics and Risk Mitigation in AI) as an engineering application of the fairness principle.

**All data is synthetic.** Both groups are generated with the same distribution of true skill and the same chance of being qualified, so any gap in errors is caused by the data pipeline or the model. Results illustrate mechanisms; they are not research findings about real people.

**Live app:** https://fairlens-ucs421.streamlit.app

## Deploy on Streamlit Community Cloud (free)

1. Push this folder to a public GitHub repository (files at the repo root: `app.py`, `fairlens_core.py`, `requirements.txt`, `.streamlit/config.toml`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Repository: `rupinderkaur1904/fairlens`, branch `main`, main file path `app.py`.
5. App URL: type `fairlens-ucs421` (so the link is `fairlens-ucs421.streamlit.app`, which the QR code on slide 8 points to).
6. Advanced settings → Python 3.12 → **Deploy**. The first build takes 2–4 minutes.

Free apps go to sleep after a period without visits. Open the link 10 minutes before presenting and click "Yes, get this app back up" if you see it.

## Run it locally

Requires Python 3.9 or newer.

```bash
cd fairlens
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. It runs fully offline once the packages are installed.

Backup without a browser (prints every table and writes `results/`):

```bash
python run_audit.py
```

## Files

| File | Purpose |
|---|---|
| `fairlens_core.py` | Synthetic data generators, both experiments, per-group metrics |
| `app.py` | Streamlit dashboard |
| `run_audit.py` | Headless run; saves CSVs, `results.json` and console output |
| `results/` | Outputs from the run used for the slides |

## Dataset schema

`qualification_score` (0–100), `years_experience`, `technical_score` (0–100), `group` (A/B), `qualified` (0/1 ground truth). Ground truth depends only on a latent skill drawn identically for both groups.

## Experiment A: dataset representation bias

Group B candidates are graded on a stricter qualification scale (about 12 points lower for the same skill) and have fewer years of formal experience. Model: gradient boosting on the three score features (group is not an input). Test set: balanced, 1,500 per group.

| Training data | Group A recall | Group B recall | Recall gap | Overall accuracy |
|---|---|---|---|---|
| Skewed (5% Group B) | 77.6% | 34.4% | 43.2 pts | 78.5% |
| Representative (50% Group B) | 76.7% | 71.5% | 5.2 pts | 82.3% |

The slider in the sidebar changes the Group B share of the skewed training set.

## Experiment B: algorithmic bias with fixed data

Balanced data (equal sizes, skill and qualification rate). Group B records about 2 fewer years of experience for the same skill (career breaks). Training, validation and test data are identical across the three designs; only the algorithm changes. Model: logistic regression.

| Design | Overall accuracy | Recall A / B | Recall gap | FPR gap |
|---|---|---|---|---|
| B1 Baseline: all features, one accuracy-optimal threshold (0.50) | 81.4% | 88.0% / 64.5% | 23.5 pts | 15.5 pts |
| B2 Drop the proxy feature (experience) | 78.2% | 68.5% / 69.6% | −1.1 pts | 0.3 pts |
| B3 Group thresholds for equal opportunity (A 0.64, B 0.37) | 82.3% | 76.5% / 73.6% | 2.9 pts | 2.2 pts |

B3 is post-processing in the spirit of Hardt, Price & Srebro (2016). It uses the group attribute at decision time, which may be restricted in real hiring.

## Metrics

Per group: accuracy, precision, recall/TPR, FPR, FNR, selection rate. Gaps: TPR (equal opportunity), FPR, FNR, selection rate (demographic parity).

## Limitations

Synthetic data with designed effects; two groups only; single seed (no confidence intervals); labels assumed correct; no intersectional or individual-fairness analysis; a demonstration of auditing, not evidence that any system is fair.
