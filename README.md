# 🏍️ MotoGP Lap Time Predictor (Burnout 2025 Finalist)

A regression project developed during the **Burnout 2025 Datathon**, organized by IEEE Computer Society MUJ. The goal was to predict MotoGP rider lap times using machine learning models on a rich dataset of racing statistics.

## 📌 Project Overview

This project was built as part of a 12-hour national-level data science competition hosted on **Kaggle**, with 124 participating teams. The competition involved exploratory data analysis, feature engineering, predictive modeling, and presenting findings in a final round with a judging panel.

Our team secured a **Top 15 finish** and was selected for **Round 2**, which involved a live 10-slide presentation and Q&A with judges.

## 🔍 Problem Statement

Predict the **Lap Time (in seconds)** for MotoGP riders using a dataset containing:

- Rider stats
- Lap-by-lap performance
- Weather conditions
- Track characteristics
- Team & strategy data

The evaluation metric was **Root Mean Squared Error (RMSE)**.

## 🚀 Highlights

- Built regression models using **XGBoost** and **Random Forest**
- **Hyperparameter tuning** with 50+ **Optuna trials**
- Applied **cross-validation** and RMSE evaluation
- Created a 10-slide presentation and live demo in Round 2
- Answered judge questions in a real-time technical Q&A

## 📁 Project Structure

- `model.py`: Full pipeline: preprocessing, model training (RF & XGBoost), Optuna tuning, evaluation, and submission generation
- `new.py`: Inference script for generating predictions on test data using the trained model
- `best_xgb_model.joblib`: Final XGBoost model saved after Optuna tuning

## 📊 Kaggle Notebook

👉 [Click here to view the notebook](https://www.kaggle.com/code/narendersingh007/notebook0a56537ec2)

## 🧠 Tools & Libraries

- Python
- Pandas, NumPy
- XGBoost, Random Forest (from scikit-learn)
- Optuna (for tuning)
- Matplotlib, Seaborn
- Jupyter Notebook

## 🏆 Competition Details

**Event:** Burnout 2025 – MotoGP Data Analytics Challenge  
**Organizer:** IEEE Computer Society MUJ  
**Platform:** Kaggle  
**Duration:** 12 hours  
**Participants:** 124 teams  
**Result:** Top 15 finish + Final round presentation

## ✅ Submission Deliverables

- `solution.csv` file with lap time predictions
- Full `Jupyter Notebook` with EDA, feature engineering, and modeling
- 10-slide presentation summarizing the approach and insights
