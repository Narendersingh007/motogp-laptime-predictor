import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("best_xgb_model.joblib")

# Load test data
test = pd.read_csv("test.csv")
original_test = test.copy()

# Drop columns NOT used during training
drop_cols = [
    'Unique ID', 'Rider_ID', 'rider', 'team', 'bike', 'shortname',
    'circuit_name', 'rider_name', 'team_name', 'bike_name'
]
test = test.drop(columns=[col for col in drop_cols if col in test.columns], errors='ignore')

# Fill missing values (same as training)
for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = test[col].fillna(test[col].mode()[0])
    else:
        test[col] = test[col].fillna(test[col].median())

# Encode categorical features (same as training)
categorical_cols = test.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col].astype(str))

# Ensure the column order matches training data exactly
expected_columns = [
    'category_x', 'Circuit_Length_km', 'Laps', 'Grid_Position', 'Avg_Speed_kmh',
    'Track_Condition', 'Humidity_%', 'Tire_Compound_Front', 'Tire_Compound_Rear',
    'Penalty', 'Championship_Points', 'Championship_Position', 'Session',
    'year_x', 'sequence', 'position', 'points', 'Corners_per_Lap',
    'Tire_Degradation_Factor_per_Lap', 'Pit_Stop_Duration_Seconds',
    'Ambient_Temperature_Celsius', 'Track_Temperature_Celsius', 'weather',
    'track', 'air', 'ground', 'starts', 'finishes', 'with_points', 'podiums',
    'wins', 'min_year', 'max_year', 'years_active'
]

test = test[expected_columns]  # Reorder to match training

# Predict
preds = model.predict(test)

# Save submission
sample = pd.read_csv("sample_submission.csv")
sample["Lap_Time_Seconds"] = preds
sample.to_csv("submission.csv", index=False)
print("  saved as 'submission.csv'")