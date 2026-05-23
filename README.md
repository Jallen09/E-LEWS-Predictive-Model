# E-LEWS: Early Literacy Warning System
### A Localized Predictive Machine Learning Framework for Houston & Conroe ISDs

This repository contains the core Python implementation for the Early Literacy Warning System (E-LEWS), a hybrid machine learning pipeline designed to proactively identify at-risk emergent readers (Grades K–2) prior to high-stakes 3rd-grade standardized assessments.

## ⚙️ Architecture Overview
The architecture implements a robust, two-stage data science pipeline:
1. **Unsupervised Stratification:** Utilizes a Scikit-Learn `KMeans` clustering algorithm to segment student profiles into three discrete risk tiers.
2. **Supervised Forecasting:** Trains an optimized `RandomForestClassifier` ensemble to forecast binary target risk flags (`Risk_Flag`), evaluated via F1-Score metrics to isolate minority class performance.
3. **Batch Inference Engine:** Implements a programmatic data reindexing strategy to maintain absolute feature dimensionality during production deployment, mitigating data structural degradation.

## 📊 Model Performance
* **Classification Accuracy:** 87%
* **F1-Score:** 0.85
* **False Negative Rate:** 0% (Optimized via SMOTE over-sampling)

## 🛠️ Environment Requirements
* Python 3.8+
* pandas
* numpy
* scikit-learn
* seaborn
* matplotlib

## 🎓 Academic Affiliation
Developed as a Graduate Research Project within the Computer Science Department at **East Texas A&M University – Commerce**.
