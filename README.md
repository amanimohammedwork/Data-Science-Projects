# Data Science & Machine Learning Portfolio

A collection of end-to-end machine learning projects covering **tabular ML, time-series forecasting, NLP, and computer vision**, applied to problems in banking, credit risk, energy, customer experience, and agriculture.

Each project is a self-contained Jupyter notebook: data exploration → feature engineering → model training and tuning → evaluation → interpretation.

---

## Projects at a glance

| # | Project | Domain | Task | Key techniques | Headline result |
|---|---------|--------|------|----------------|-----------------|
| 1 | [Bank Customer Churn](#1-bank-customer-churn-prediction) | Banking | Binary classification | XGBoost, Logistic Regression, SHAP | XGBoost: **F1 0.60**, ROC-AUC **0.87** |
| 2 | [Credit Default Risk](#2-credit-default-risk-home-credit) | Credit risk | Binary classification | Feature engineering, XGBoost, Random Forest, Keras NN, SHAP | XGBoost: ROC-AUC **0.762**, KS **0.39** |
| 3 | [Electricity Demand Forecasting](#3-electricity-demand-forecasting-denmark) | Energy | Time-series regression | XGBoost, LightGBM, TimeSeriesSplit, SHAP | **19.8% lower MAE** than ENTSO-E day-ahead forecast |
| 4 | [Negative Customer Feedback Classification](#4-negative-customer-feedback-classification-danish) | NLP | Multi-class text classification | Zero-shot NLI, Llama 3.1 (4-bit), few-shot prompting, BERT fine-tuning | Few-shot Llama 3.1: **66% accuracy** |
| 5 | [Plant Disease Recognition](#5-plant-disease-recognition) | Computer vision | 38-class image classification | Custom CNN, ResNet-18, MobileNetV2, CLIP ViT, autoencoder | Best fine-tuned models: **~94.7%** validation accuracy |

---

## 1. Bank Customer Churn Prediction

📓 `Banking.ipynb`

**Goal:** Identify which bank customers are likely to leave, and why, so the bank can target retention efforts before revenue is lost.

**Data:** [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) (Kaggle), with customer age, gender, country, tenure, balance, credit score, number of products, and active-member status. About 20% of customers churned.

**Approach**
- Exploratory analysis segmented by gender, age, country, tenure, balance, salary and credit score, plus a correlation matrix.
- Compared a tree-based model (**XGBoost**) against a linear baseline (**Logistic Regression**), both wrapped in scikit-learn pipelines and tuned with 5-fold `GridSearchCV` optimising **F1**.
- Explained the best model with **SHAP**.

**Results (held-out test set)**

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| **XGBoost** | 87.1% | 76.0% | 49.9% | **60.2%** |
| Logistic Regression | 72.8% | 39.5% | 72.5% | 51.2% |

XGBoost reaches an ROC-AUC of 0.87. Logistic Regression catches more churners (higher recall) but at a much higher false-alarm rate.

**Key findings**
- Older customers (40–60) churn at a much higher rate; age has the strongest correlation with churn (0.29).
- Germany has a noticeably higher churn share than France or Spain.
- Inactive members and customers with higher balances are more likely to leave.
- Estimated salary has almost no visible impact on churn.
- SHAP ranks **age**, **number of products**, and **active-member status** as the top three drivers.

---

## 2. Credit Default Risk (Home Credit)

📓 `Risk_mangement.ipynb`

**Goal:** Predict the probability that a loan applicant will default, and build the tooling a risk team would want around such a model: calibration, subgroup checks, applicant-level explanations, and what-if analysis.

**Data:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle competition): the main application table plus bureau, previous applications, installment payments, POS-cash and credit-card tables.

**Approach**
- **Cleaning:** dropped columns with >65% missing, median/"Missing" imputation for 30–65%, fixed the `365243` sentinel in `DAYS_EMPLOYED`, converted day counts to years, added missing-value indicator flags.
- **Feature engineering:** ratio features (credit-to-income, annuity-to-income, credit-to-goods-price, employment-to-age, income per family member, and others), plus per-customer aggregations of the six supplementary tables.
- **Models:** Logistic Regression, Random Forest, XGBoost, and a Keras neural network, tuned with 3-fold `GridSearchCV` on ROC-AUC (stratified 80/20 split).
- **Evaluation and governance:** ROC-AUC, PR-AUC, KS statistic, calibration curve, predicted-probability distribution, SHAP summary and waterfall plots, ROC-AUC by subgroup (gender, age, income, education, housing, marital status, occupation), and a what-if analysis of default risk vs. income.

**Results (validation set)**

| Model | ROC-AUC | PR-AUC | KS |
|-------|--------:|-------:|---:|
| **XGBoost** | **0.762** | **0.242** | **0.394** |
| Neural Network | 0.748 | 0.227 | 0.366 |
| Logistic Regression | 0.747 | 0.226 | 0.370 |
| Random Forest | 0.745 | 0.218 | 0.372 |

> The dataset is highly imbalanced (~8% defaults). At the default 0.5 threshold XGBoost reaches 92% accuracy but only ~3% recall, so in practice the decision threshold should be chosen from the business cost of a missed default vs. a rejected good customer.

---

## 3. Electricity Demand Forecasting (Denmark)

📓 `Demand_Forecasting.ipynb`

**Goal:** Measure how accurate the official ENTSO-E day-ahead load forecast is for Denmark's two bidding zones (DK1 and DK2), find out when forecasting is hardest, and test whether an ML model can beat it.

**Data:** Hourly *Actual Total Load* and *Day-ahead Total Load Forecast* (MW) from [ENTSO-E](https://transparency.entsoe.eu/), covering 1 Jan – 24 Aug 2026.

**Approach**
- Parsed CET/CEST timestamps, separated real gaps from future hours that have a forecast but no actual yet, and linearly interpolated the real gaps.
- Analysed load by hour, weekday/weekend, day of week, month and season, plus forecast error by hour, day, month and Danish public holidays.
- Built naive baselines (previous hour / day / week) and benchmarked the ENTSO-E forecast.
- Trained **XGBoost** and **LightGBM** using calendar features and lagged load, tuned with `TimeSeriesSplit` (5 folds) on a chronological 80/20 split. Interpreted the model with SHAP.
- Segmented model error by demand level (quartiles) to see where the model is weakest.

**Results (test period)**

| Forecast | MAE (MW) | RMSE (MW) | 95th pct. abs. error (MW) |
|----------|---------:|----------:|--------------------------:|
| Previous-week baseline | 239.5 | 325.9 | 675.3 |
| Previous-day baseline | 190.8 | 268.6 | 601.4 |
| Previous-hour baseline | 97.5 | 129.0 | 258.1 |
| ENTSO-E day-ahead | 71.2 | 127.5 | 283.2 |
| **XGBoost** | **57.1** | **82.0** | **163.1** |

XGBoost improves on ENTSO-E by **19.8% (MAE)**, **35.7% (RMSE)** and **42.4% (95th-percentile error)**. SHAP shows the previous-hour load is the dominant predictor, followed by hour of day.

**Key findings**
- Demand is highest in winter and on weekdays (peaking midweek), and lowest on Sundays and in summer.
- Aggregate improvement hides differences by regime: ENTSO-E stays more consistent than XGBoost in the highest-demand quartile.

---

## 4. Negative Customer Feedback Classification (Danish)

📓 `NegativeCustomerFeedback.ipynb`

**Goal:** Automatically sort low-rated Danish customer reviews into the type of complaint, so a business can see what to fix first.

**Data:** 100 randomly sampled 1–2 star reviews from the [Danish reviews dataset](https://github.com/AlessandroGianfelici/danish_reviews_dataset), hand-labelled into six categories:
*Credibility (Troværdighed), Product quality (Produktkvalitet), Customer service (Kundeservice), Pricing (Prissætning), Delivery issues (Leveringsproblemer), Return/refund issues (Retur og refundering).*

**Approaches compared**
1. **Zero-shot NLI:** `joeddav/xlm-roberta-large-xnli`
2. **Zero-shot LLM:** `Meta-Llama-3.1-8B-Instruct` (4-bit quantised, runs on ~8 GB VRAM), Danish prompt
3. **Few-shot LLM:** same model with 5 labelled examples in the prompt
4. **Fine-tuned `bert-base-multilingual-cased`:** with oversampling to balance classes
5. **Fine-tuned `danish-bert-botxo`:** Danish-specific BERT, same setup

**Results (macro-averaged)**

| Approach | Precision | Recall | F1 | Accuracy |
|----------|----------:|-------:|---:|---------:|
| NLI zero-shot | 0.639 | 0.413 | 0.375 | 49% |
| LLM zero-shot | 0.383 | 0.373 | 0.282 | 42% |
| **LLM few-shot** | **0.738** | **0.670** | **0.649** | **66%** |
| BERT fine-tuned (multilingual) | 0.315 | 0.454 | 0.371 | 48% |
| BERT fine-tuned (Danish) | 0.407 | 0.447 | 0.339 | 48% |

**Takeaway:** With only 100 labelled examples, showing a few examples to a capable LLM beat both zero-shot approaches and the fine-tuned BERT models. Fine-tuning likely needs considerably more labelled data. Note that the BERT models are scored on a 25-review test split, while the other three approaches are scored on all 100, so the comparison is indicative rather than strict.

> ⚠️ Requires a GPU, plus access to the gated Llama 3.1 weights on Hugging Face.

---

## 5. Plant Disease Recognition

📓 `PlantDiseaseRecognition.ipynb`

**Goal:** Classify leaf images into 38 crop-disease/healthy classes, and compare model families on accuracy and efficiency.

**Data:** [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (Kaggle): 256×256 RGB leaf images, ~70k training and ~17.6k validation images.

**What's covered**
- **Data understanding:** sample visualisation, class balance, resolution, mean colour channels, and a Laplacian-variance sharpness check (~17% of train/validation images fall below the sharpness threshold).
- **Custom CNN** in PyTorch: three conv blocks with batch norm, dropout, adaptive pooling; Adam vs. SGD comparison.
- **Transfer learning:** ResNet-18 and MobileNetV2, each **frozen** (train the head only) vs. **partially fine-tuned**.
- **Vision Transformer:** CLIP ViT-L/14 with a trained classification head.
- **Convolutional autoencoder** to test whether compressed latent features support classical classifiers (Random Forest, logistic regression, feed-forward network).
- **Interpretability:** SHAP image explanations of which leaf regions drive predictions.

**Validation accuracy**

| Model | Frozen backbone | Fine-tuned |
|-------|----------------:|-----------:|
| ResNet-18 | 93.11% | 94.70% |
| MobileNetV2 | 94.56% | **94.73%** |
| CLIP ViT-L/14 (head only) | 93.76% | n/a |

**Takeaway:** Lightweight pretrained models get to ~94–95% quickly. MobileNetV2 matches or beats ResNet-18 while being far smaller, and fine-tuning adds only a small gain over frozen features.

> 🚧 **Status:** the custom-CNN tuning comparison and the autoencoder-downstream-classifier section are still in progress.

---

## Tech stack

- **Language:** Python 3
- **Data & analysis:** pandas, NumPy, SciPy, Matplotlib, Seaborn, Plotly
- **Classical ML:** scikit-learn, XGBoost, LightGBM
- **Deep learning:** PyTorch, torchvision, TensorFlow/Keras
- **NLP & LLMs:** Hugging Face Transformers, Datasets, `bitsandbytes` (4-bit quantisation)
- **Computer vision:** OpenCV, Pillow
- **Explainability:** SHAP
- **Other:** `holidays` (Danish public holidays), `kagglehub` (dataset download)

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create an environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy scipy matplotlib seaborn plotly scikit-learn \
            xgboost lightgbm shap holidays kagglehub jupyter \
            torch torchvision tensorflow opencv-python pillow \
            transformers datasets evaluate accelerate bitsandbytes

# 3. Launch Jupyter and open a notebook
jupyter lab
```

**Data access**
- **Kaggle datasets** (Banking, Risk, Plant Disease): notebooks download via `kagglehub`. You'll need a free Kaggle account, and for the Home Credit competition you must first accept the rules on Kaggle.
- **Demand forecasting:** download the DK1 and DK2 *Total Load – Day Ahead / Actual* CSVs from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and save them as `DK1.csv` and `DK2.csv` next to the notebook.
- **Danish reviews:** loaded directly from GitHub inside the notebook.

**Hardware:** the NLP and computer-vision notebooks are designed for a GPU (Google Colab or a local CUDA card). The tabular and forecasting notebooks run fine on a laptop CPU.

## Repository structure
```
├── Banking.ipynb                    # Customer churn prediction,
├── Risk_mangement.ipynb             # Credit default risk modelling
├── Demand_Forecasting.ipynb         # Electricity load forecasting (Denmark)
├── NegativeCustomerFeedback.ipynb   # Danish review complaint classification
├── PlantDiseaseRecognition.ipynb    # Leaf disease image classification
└── README.md

https://www.linkedin.com/in/amanimohammedwork/

