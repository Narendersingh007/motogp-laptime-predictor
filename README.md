# 🏍️ MotoGP Lap Time Predictor (Burnout 2025 Finalist)

A **Streamlit web app + ML pipeline** developed during the **Burnout 2025 Datathon** organized by IEEE Computer Society MUJ.  
The project predicts **MotoGP rider lap times** and provides an **interactive analytics dashboard** to explore race data.  

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motogp-laptime-predictor.streamlit.app)

---

## 📌 Project Overview  

- Built during a **12-hour national-level data science hackathon** hosted on Kaggle  
- **124 teams** participated — our team secured a **Top 15 finish** and reached **Round 2 (Finals)**  
- Workflow included **EDA → Feature Engineering → Predictive Modeling → Hyperparameter Tuning → Live Presentation**  
- Deployment-ready Streamlit app for **real-time predictions & analytics**  

---

## 🔍 Problem Statement  

Predict the **Lap Time (in seconds)** for MotoGP riders using:  
- Rider stats & experience  
- Track characteristics  
- Weather & environmental conditions  
- Tire compounds & race strategy  

**Evaluation metric:** Root Mean Squared Error (RMSE)  

---

## 🚀 Key Highlights  

- ⚡ **Models**: XGBoost (primary) + Random Forest (backup)  
- 🎯 **Tuning**: 50+ Optuna trials for hyperparameter optimization  
- 📊 **Analytics**: Lap time distributions, weather/temperature impacts, correlation heatmaps, feature importance, category-wise comparisons  
- 🖥️ **App**: Streamlit dashboard with prediction form + analytics + model insights  
- 🏆 **Competition**: Live 10-slide presentation + real-time Q&A with judges  

---

## 📂 Project Structure  
```
motogp-laptime-predictor/
│── app.py              # Main Streamlit app
│── requirements.txt    # Dependencies
│── setup.py            # Install/setup file
│── data/               # Training/test datasets (train.csv etc.)
│── models/             # Trained ML models (XGBoost .joblib, etc.)
│── utils/              # Helper scripts (imports, fixes, model utils)
│── notebooks/          # Jupyter notebooks (EDA, experiments)
│── .streamlit/         # Streamlit config
```
---

## ⚡ Streamlit App Features  

### 🔮 **Lap Time Prediction**  
Interactive form to input rider, track, and weather details → Predict lap times with confidence scores  

### 📈 **Analytics Dashboard**  
- Lap Time distribution histograms  
- Weather vs Lap Time impact (boxplots)  
- Grid Position vs Lap Time scatter  
- Temperature correlation plots  
- Correlation heatmap (numeric features)  
- Feature importance ranking  
- Violin plots by race category  

**🧾 Inferences** are displayed directly under each visualization for storytelling.  

---

## 🧠 Tools & Libraries  

- **Python**: Pandas, NumPy  
- **ML**: XGBoost, Scikit-learn, Random Forest  
- **Optimization**: Optuna  
- **Visualization**: Plotly, Matplotlib, Seaborn  
- **Deployment**: Streamlit  

---

## 🏆 Competition Details  

- **Event:** Burnout 2025 – MotoGP Data Analytics Challenge  
- **Organizer:** IEEE Computer Society MUJ  
- **Platform:** Kaggle  
- **Duration:** 12 hours  
- **Participants:** 124 teams  
- **Result:** Top 15 finish + Finals presentation  

---



## 🔧 Run Locally  
```bash
git clone https://github.com/Narendersingh007/motogp-laptime-predictor.git
cd motogp-laptime-predictor
pip install -r requirements.txt
streamlit run app.py
```
## 🧪 Testing
- Notebooks in `notebooks/` demonstrate EDA, experiments, and model validation
- Visualizations and metric evaluations can be run interactively

---

## 📜 License
- This project is licensed under **MIT License** – see [LICENSE](LICENSE) for details

---

## 🤝 Contributing
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

---

## 📫 Contact
- **Author:** Narender Singh
- **GitHub:** [Narendersingh007](https://github.com/Narendersingh007)


