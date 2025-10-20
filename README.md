# 🏍️ MotoGP Lap Time Predictor (Burnout 2025 Finalist)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motogp-laptime-predictor.streamlit.app)

A **Streamlit web app + ML pipeline** that predicts MotoGP rider lap times using a fine-tuned XGBoost model. This project was developed for the **Burnout 2025 Datathon** and features real-time predictions and an interactive analytics dashboard.

---

## 🚀 App Demo

The live application provides two main features: a real-time prediction form and an analytics dashboard powered by the training data.

| Prediction Form | Real-Time Analytics |
| :---: | :---: |
| **Users can input race parameters to get a live prediction.** | **The dashboard visualizes real data on feature importances, correlations, and distributions.** |
| <img width="1709" height="886" alt="Screenshot 2025-10-20 at 7 03 56 PM" src="https://github.com/user-attachments/assets/56e0712f-d6d4-4d07-9795-18db65823574" />| <img width="1709" height="888" alt="Screenshot 2025-10-20 at 7 04 53 PM" src="https://github.com/user-attachments/assets/6f2992d8-1d67-4476-94ab-6dda398eed1c" />

 ---

## 📂 Project Structure

The project is organized to separate the Streamlit app, training notebooks, and helper utilities.
```
motogp-laptime-predictor/
├── README.md               
├── app.py                  # The main Streamlit web application
├── requirements.txt        # All Python packages needed to run the app
│
├── data/
│   └── train_sample.csv    # The small sample used by the app's analytics tab
│
├── notebooks/
│   └── from-raw-telemetry-to-optimized-models-motogp-lap.ipynb         
│
├── utils/
│   └── model.py            # The Python script used to train the final model
│
├── .gitignore              
└── LICENSE               
```
---

## 🔧 Run Locally

To run this application on your local machine, follow these steps.

**Note:** The application is designed to download the pre-trained model (`.joblib` file) from the project's GitHub Releases page on its first run.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Narendersingh007/motogp-laptime-predictor.git](https://github.com/Narendersingh007/motogp-laptime-predictor.git)
    cd motogp-laptime-predictor
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python3 -m venv motogp-env
    source motogp-env/bin/activate  # On Windows: motogp-env\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The app will open in your browser. On the first launch, it will display a message "Downloading model..." while it fetches the artifact.

---
## ⚡ Streamlit App Features  

### 🔮 **Lap Time Prediction** Interactive form to input rider, track, and weather details → Predict lap times using the real, pre-trained XGBoost model.

### 📈 **Analytics Dashboard** - **Real Data:** All plots are generated from a `train_sample.csv` of the original dataset.
- **Real Model Importances:** The feature importance plot is generated directly from the loaded model.
- Lap Time distribution histograms  
- Weather vs Lap Time impact (boxplots)  
- Grid Position vs Lap Time scatter  
- Temperature correlation plots  
- Correlation heatmap (numeric features)  
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
