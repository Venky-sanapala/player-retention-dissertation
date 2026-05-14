# Player Retention & Difficulty Progression in Educational Programming Games
### A Machine Learning and Data Analytics Approach

**MSc Data Science Dissertation** — Venkatesh Sanapala  
Aston University, Birmingham | Module: AM41PR | 2024–2025

---

## Overview

This repository contains the full analytical pipeline for an MSc dissertation investigating how difficulty progression influences player retention in educational programming games. The study uses a synthetic dataset of 1,200 player records to build churn prediction models, segment players by behaviour, and simulate an adaptive difficulty engine grounded in flow theory.

---

## Research Questions

1. What behavioural and demographic factors most strongly predict player churn?
2. How well do difficulty progression patterns predict performance and retention?
3. Can unsupervised learning reliably identify distinct player segments with meaningfully different churn risks?
4. Does a rule-based adaptive difficulty engine reduce performance variance and sustain flow-zone engagement across diverse player profiles?

---

## Key Findings

- **Completion rate** is the single strongest predictor of retention (retained mean = 0.576 vs churned mean = 0.254)
- **Flow score** is the strongest continuous predictor of retention (r = 0.396, p < 0.0001)
- Difficulty jumps of 3+ levels significantly degrade performance and accelerate churn
- Best churn prediction model (Logistic Regression) achieved **AUC = 0.973**
- K-Means clustering identified **4 behavioural segments** with churn rates ranging from 0% to 24.3%
- The adaptive difficulty engine converged all player profiles into the flow zone **within 5 sessions**

---

## Repository Structure

```
player-retention-dissertation/
│
├── dissertation_pipeline.py     # Full analytical pipeline (9 sequential steps)
├── player_dataset.csv           # Synthetic dataset (1,200 players, 25 features)
│
├── outputs/
│   ├── figures/                 # All 9 generated figures (distributions, ROC curves, etc.)
│   └── csv/                     # Supporting CSV outputs from pipeline steps
│
└── README.md
```

---

## Pipeline Steps

The pipeline (`dissertation_pipeline.py`) runs 9 sequential steps:

1. Dataset generation using NumPy random sampling
2. Descriptive statistics computation
3. Univariate and bivariate distribution visualisation
4. Churn and retention analysis
5. Difficulty progression analysis
6. Correlation matrix and heatmap visualisation
7. Inferential statistical testing (t-tests, ANOVA, Pearson correlation)
8. K-Means clustering with elbow method selection and PCA visualisation
9. Supervised ML for churn prediction + optimal difficulty classification
10. Adaptive difficulty engine simulation (20-session horizon, 3 player profiles)

---

## Methods

### Dataset
- 1,200 synthetic player records with 25 features across 6 domains: demographics, session engagement, difficulty progression, error & help-seeking behaviour, social & achievement engagement, and derived performance scores
- Churn rate: 3.2% (realistic platform conditions)
- All random seeds fixed at `numpy.random.seed(42)` for full reproducibility

### Machine Learning Models

| Model | AUC | Accuracy | F1-Score | CV-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.973 | 0.971 | 0.364 | 0.968 |
| Gradient Boosting (XGB) | 0.959 | 0.954 | 0.154 | 0.914 |
| Random Forest | 0.950 | 0.958 | 0.167 | 0.945 |

### Player Segments (K-Means, k=4)

| Segment | Completion Rate | Flow Score | Hint Rate | Churn % |
|---|---|---|---|---|
| High Performers | 0.702 | 81.6 | 0.07 | 0.0% |
| Engaged Learners | 0.669 | 74.9 | 0.13 | 0.3% |
| Casual Players | 0.352 | 62.2 | 0.10 | 5.2% |
| Struggling Beginners | 0.447 | 44.5 | 0.76 | 24.3% |

### Adaptive Difficulty Engine
A rule-based system adjusting difficulty each session based on a composite engagement score (completion rate × 0.40 + attempt efficiency × 0.35 + hint avoidance × 0.25):
- Score > 75 → increase difficulty by 1 level
- Score < 45 → decrease difficulty by 1 level
- Score 45–75 → maintain current difficulty

---

## Theoretical Framework

- **Flow Theory** (Csikszentmihalyi, 1990) — optimal challenge calibration
- **Self-Determination Theory** (Ryan & Deci, 2000) — intrinsic vs extrinsic motivation
- **Educational Data Mining** (Baker & Inventado, 2014) — behavioural proxies for psychological states

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost
```

---

## Usage

```bash
# Clone the repository
git clone https://github.com/Venky-sanapala/player-retention-dissertation.git
cd player-retention-dissertation

# Run the full pipeline
python dissertation_pipeline.py
```

Outputs (figures and CSV files) will be saved to the `outputs/` directory.

---

## Limitations

- Synthetic data cannot fully capture real inter-player correlations (social influence, cohort dynamics)
- Rule-based adaptive engine does not model uncertainty or multi-step planning
- AUC is threshold-agnostic; production deployment would require threshold calibration against intervention cost structure
- Platform-level dynamics (content updates, seasonality) are not modelled

---

## Future Work

- Validation on real platform data (anonymised interaction logs)
- Reinforcement learning for adaptive difficulty (Markov decision process formulation)
- NLP analysis of code submission histories to detect conceptual misconceptions
- Causal modelling of social engagement's effect on retention

---

## Citation

```
Sanapala, V. (2025). Player Retention and Difficulty Progression in Educational
Programming Games: A Machine Learning and Data Analytics Approach.
MSc Data Science Dissertation, Aston University, Birmingham.
```

---

## License

This project is submitted as academic work for Aston University's MSc Data Science programme. Please contact the author before reusing any part of this work.
