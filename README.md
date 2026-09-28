# Data Science & Machine Learning Portfolio

A collection of end-to-end machine learning projects covering **medical imaging, tabular ML, time-series forecasting, NLP, and computer vision**, applied to problems in healthcare, banking, credit risk, energy, customer experience, and agriculture.

Each project is a self-contained Jupyter notebook: data exploration → feature engineering → model training and tuning → evaluation → interpretation.

---

## Projects at a glance

| # | Project | Domain | Task | Key techniques | Headline result |
|---|---------|--------|------|----------------|-----------------|
| ★ | [**Master's Thesis: Retinal Disease Analysis**](#master-thesis-cnns-vs-vision-transformers-for-retinal-disease-under-data-scarcity) | Healthcare / ophthalmology | 8-class classification + lesion segmentation | EfficientNet-B0, RETFound (ViT), U-Net, Grad-CAM, Attention Rollout, Streamlit app | RETFound: **QWK 0.965** vs 0.901; EfficientNet-B0: **Dice 0.747** vs 0.241 |
| 1 | [Bank Customer Churn](#1-bank-customer-churn-prediction) | Banking | Binary classification | XGBoost, Logistic Regression, SHAP | XGBoost: **F1 0.60**, ROC-AUC **0.87** |
| 2 | [Credit Default Risk](#2-credit-default-risk-home-credit) | Credit risk | Binary classification | Feature engineering, XGBoost, Random Forest, Keras NN, SHAP | XGBoost: ROC-AUC **0.762**, KS **0.39** |
| 3 | [Electricity Demand Forecasting](#3-electricity-demand-forecasting-denmark) | Energy | Time-series regression | XGBoost, LightGBM, TimeSeriesSplit, SHAP | **19.8% lower MAE** than ENTSO-E day-ahead forecast |
| 4 | [Negative Customer Feedback Classification](#4-negative-customer-feedback-classification-danish) | NLP | Multi-class text classification | Zero-shot NLI, Llama 3.1 (4-bit), few-shot prompting, BERT fine-tuning | Few-shot Llama 3.1: **66% accuracy** |
| 5 | [Plant Disease Recognition](#5-plant-disease-recognition) | Computer vision | 38-class image classification | Custom CNN, ResNet-18, MobileNetV2, CLIP ViT, autoencoder | Best fine-tuned models: **~94.7%** validation accuracy |

---

## Master Thesis: CNNs vs. Vision Transformers for Retinal Disease under Data Scarcity

**A Multi-Model Deep Learning Analysis of Retinal Pathologies: Comparing Convolutional Neural Networks and Vision Transformers Under Strict Data Scarcity**
Amani Hussein · Department of Mathematics and Computer Science, University of Southern Denmark (SDU) · Supervisor: Tariq Yousif · June 2026

**Goal:** Assess whether a standard CNN (**EfficientNet-B0**) or a retina-specific Vision Transformer foundation model (**RETFound**) makes the better diagnostic tool for the three most common sight-threatening retinal diseases (diabetic retinopathy, dry age-related macular degeneration and glaucoma) when only a few hundred labelled images are available. Both models are compared on **classification** and on **pixel-level lesion segmentation**.

**Research questions**
1. How do EfficientNet-B0 and RETFound perform in classification and segmentation of retinal disease when data is limited?
2. To what extent does a domain-specific ViT foundation model outperform a standard CNN in feature-representation efficiency and optimisation stability?
3. How do spatial and class imbalance distort conventional metrics, and which statistics best ensure diagnostic reliability?

**Data**
- 393 colour fundus images gathered from public sources (National Eye Institute, Mendeley Data, Kaggle) and **graded and annotated by the author, drawing on a background in optometry**.
- 8 classes: DR (mild / moderate / severe, graded on the ICDR scale), dry AMD (early / intermediate / late, Beckman classification), glaucoma (cup-to-disc ratio > 0.5) and healthy controls.
- Pixel-level annotations in Roboflow for optic cup and disc, exudates, cotton-wool spots, microaneurysms, haemorrhages, drusen, geographic atrophy, pigmentation abnormalities and imaging artefacts.
- Underrepresented classes were augmented to 100 images each (flips, rotations); further clinical-style augmentation (brightness, hue/saturation, Gaussian blur, CLAHE) was used for regularisation. 80 / 10 / 10 train / validation / test split (**39 test images**).

**Approach**
- **EfficientNet-B0 classifier:** ImageNet-pretrained, last two MBConv blocks plus `top_conv`/`top_bn` and head fine-tuned; AdamW, cosine annealing, label smoothing 0.1, early stopping.
- **RETFound classifier:** ViT-Small (patch 14) with retinal pre-trained weights; linear probing gave the best results after comparing frozen, partial and full fine-tuning.
- **EfficientNet-B0 U-Net segmenter:** BCE + Dice loss, 512×512 input.
- **RETFound segmenter:** custom convolutional decoder on the ViT encoder with positional-embedding interpolation, Focal-Tversky loss (α = 0.3, β = 0.3, γ = 2.0).
- **Evaluation beyond accuracy:** sensitivity, specificity, precision, macro-F1, AUROC, **quadratic-weighted Cohen's Kappa**, confusion matrices and **reliability diagrams** for calibration. Segmentation is scored with sensitivity, pixel precision, Dice, Jaccard and HD95.
- **Explainability:** **Grad-CAM** for the CNN and **Attention Rollout** across all 12 ViT layers, used to check *why* each model decides as it does.

**Results: classification (39-image test set; both models classify 34 / 39 correctly)**

| Metric | EfficientNet-B0 | RETFound |
|--------|----------------:|---------:|
| Sensitivity | 0.872 | 0.872 |
| Specificity | 0.981 | 0.980 |
| Macro F1 | 0.909 | 0.910 |
| AUROC | 0.967 | 0.965 |
| Quadratic-weighted Kappa | 0.901 | **0.965** |

**Results: lesion segmentation**

| Metric | EfficientNet-B0 (U-Net) | RETFound |
|--------|------------------------:|---------:|
| Sensitivity | **0.775** | 0.359 |
| Pixel precision | **0.856** | 0.224 |
| Dice (DSC) | **0.747** | 0.241 |
| Jaccard | **0.652** | 0.225 |
| HD95 (px, lower is better) | 145.3 | **120.1** |

**Key findings**
- **Headline metrics hide the difference.** The two classifiers are almost identical on sensitivity, specificity, F1 and AUROC. Kappa, which penalises multi-stage staging errors, separates them in favour of RETFound.
- **EfficientNet-B0 learned a shortcut.** Grad-CAM shows its attention on image borders and the camera mask rather than on lesions, and its reliability diagram shows overconfidence. **RETFound is underconfident** (probabilities stay below ~0.65) but its Attention Rollout locks onto the macula and lesions in deeper layers, which is the more clinically credible behaviour.
- **Error profiles differ.** EfficientNet-B0 mislabelled mild and moderate DR as healthy (missed disease). RETFound's errors leaned towards false alarms on healthy eyes, plus one missed glaucoma case.
- **The picture reverses for segmentation.** EfficientNet-B0 wins on overlap but over-bounds lesions; RETFound under-segments, sometimes predicting no mask at all.
- **A data-loading bug caught by visual inspection.** The first U-Net run scored Dice 0.994. Plotting predictions showed that colour-thresholding was extracting the whole retina as the "lesion". After re-parsing the COCO-JSON polygons, the honest score was 0.747. The lesson: sanity-check near-perfect metrics.
- **Conclusion:** RETFound is preferred for classification, and the CNN for segmentation. The author is exploring a hybrid ConViT-style model, plus larger and more diverse data, synthetic images (GANs / diffusion) and temperature scaling to fix calibration.

**Limitations**
- Small test set (39 images, 3–4 per stage), so a single image moves a per-class score by 25–33 points. Treat differences as indicative.
- Public-source images with no patient metadata; low-quality images were excluded, which may reduce real-world robustness.

**Clinical decision-support demo (Streamlit, "AEye").** A live inference interface built for optometrists, showing classification, segmentation and explainability views, with **stress-test sliders (brightness, contrast, blur)** for checking robustness to imaging variation.

> This is a research prototype. It is **not a certified medical device** and must not be used for real diagnosis.

---

## 1. Bank Customer Churn Prediction

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

>  Requires a GPU, plus access to the gated Llama 3.1 weights on Hugging Face.

---

## 5. Plant Disease Recognition

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

---

## Tech stack

- **Language:** Python 3
- **Data & analysis:** pandas, NumPy, SciPy, Matplotlib, Seaborn, Plotly
- **Classical ML:** scikit-learn, XGBoost, LightGBM
- **Deep learning:** PyTorch, torchvision, timm, torchseg (U-Net), TensorFlow/Keras
- **NLP & LLMs:** Hugging Face Transformers, Datasets, `bitsandbytes` (4-bit quantisation)
- **Computer vision & medical imaging:** OpenCV, Pillow, Albumentations, Roboflow (annotation), MedPy (HD95), pycocotools
- **Explainability:** SHAP, Grad-CAM, Attention Rollout
- **Apps & other:** Streamlit, `holidays` (Danish public holidays), `kagglehub`

https://www.linkedin.com/in/amanimohammedwork/

