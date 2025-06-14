import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor, plot_importance
import optuna
import os

FAST_RUN = False  # Set to True to use smaller data for quick testing

# --------------------- Timing Utilities ---------------------

def timeit(label):
    print(f"\n{label}...", end="")
    return time.time(), label

def done(t0, label):
    print(f" Done in {time.time() - t0:.2f} sec [{label}]")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --------------------- Load Data ---------------------

t0, label = timeit("Loading data")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
val = pd.read_csv('val.csv')
done(t0, label)

if FAST_RUN:
    train = train.sample(5000, random_state=42)
    test = test.sample(1000, random_state=42)
    val = val.sample(1000, random_state=42)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Val shape:", val.shape)

# --------------------- Preprocessing ---------------------

t0, label = timeit("Dropping unnecessary columns")
drop_cols = ['Unique ID', 'Rider_ID', 'rider', 'team', 'bike', 
             'shortname', 'circuit_name', 'rider_name', 'team_name', 'bike_name']
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True, errors='ignore')
val.drop(columns=drop_cols, inplace=True, errors='ignore')
done(t0, label)

t0, label = timeit("Handling missing values")
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_cols = train.select_dtypes(include=np.number).drop(columns=['Lap_Time_Seconds']).columns.tolist()
cat_cols = train.select_dtypes(include='object').columns.tolist()

train[num_cols] = num_imputer.fit_transform(train[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])
val[num_cols] = num_imputer.transform(val[num_cols])

train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])
val[cat_cols] = cat_imputer.transform(val[cat_cols])
done(t0, label)

t0, label = timeit("Encoding categorical variables")
for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col], val[col]], axis=0).astype(str)
    le.fit(all_vals)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    val[col] = le.transform(val[col].astype(str))
done(t0, label)

# --------------------- Feature and Target ---------------------

X = train.drop('Lap_Time_Seconds', axis=1)
y = train['Lap_Time_Seconds']
X_val_hold = val.drop('Lap_Time_Seconds', axis=1)
y_val_hold = val['Lap_Time_Seconds']

t0, label = timeit("Splitting train and validation sets")
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y_bins, random_state=42)
done(t0, label)

# --------------------- Random Forest Model ---------------------

t0, label = timeit("Training Random Forest")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
done(t0, label)

t0, label = timeit("Evaluating Random Forest")
rf_preds = rf_model.predict(X_val)
print(f"Random Forest RMSE: {rmse(y_val, rf_preds):.4f}, R²: {r2_score(y_val, rf_preds):.4f}")
done(t0, label)

# --------------------- Baseline Dummy Model ---------------------

t0, label = timeit("Training Dummy Regressor")
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
print(f"Dummy Regressor RMSE: {rmse(y_val, dummy.predict(X_val)):.4f}")
done(t0, label)

# --------------------- Optuna for XGBoost ---------------------

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
    }

    model = XGBRegressor(
        **params,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return rmse(y_val, preds)

t0, label = timeit("Tuning XGBoost")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
done(t0, label)

print("Best XGBoost parameters found:", study.best_params)

# --------------------- Final Model Training ---------------------

best_model = XGBRegressor(
    **study.best_params,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42
)
best_model.fit(X_train, y_train)

joblib.dump(best_model, 'best_xgb_model.joblib')
print("Model saved to 'best_xgb_model.joblib'")

t0, label = timeit("Evaluating XGBoost")
xgb_preds = best_model.predict(X_val)
print(f"XGBoost RMSE: {rmse(y_val, xgb_preds):.4f}, R²: {r2_score(y_val, xgb_preds):.4f}")
done(t0, label)

plot_importance(best_model, max_num_features=10)
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# --------------------- Cross-Validation ---------------------

if not FAST_RUN:
    t0, label = timeit("Performing Cross-validation")
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    print("Cross-validation RMSE scores:", -cv_scores)
    print("Average RMSE:", -cv_scores.mean())
    done(t0, label)

# --------------------- Submission ---------------------

t0, label = timeit("Generating predictions for test set")
test_preds = best_model.predict(test)
done(t0, label)

t0, label = timeit("Saving submission file")
sample = pd.read_csv('sample_submission.csv')
sample['Lap_Time_Seconds'] = test_preds
sample.to_csv('submission.csv', index=False)
done(t0, label)

print("Submission saved as 'submission.csv'")