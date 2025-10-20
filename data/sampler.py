import pandas as pd
import os

# --- Configuration ---
FULL_DATA_PATH = "train.csv"         # The big file (it's in the same folder)
SAMPLE_DATA_PATH = "train_sample.csv"  # The new file we will create
SAMPLE_SIZE = 1500                   # How many rows to keep
# ---------------------

print(f"Loading full dataset from {FULL_DATA_PATH}...")

# Check if the file exists
if not os.path.exists(FULL_DATA_PATH):
    print(f"ERROR: Cannot find {FULL_DATA_PATH}")
    print("Please make sure 'train.csv' is in the same 'data' folder as this script.")
else:
    # Load your full dataset
    full_data = pd.read_csv(FULL_DATA_PATH)

    print(f"Successfully loaded {len(full_data)} rows.")

    # Make sure we don't ask for more rows than we have
    if len(full_data) < SAMPLE_SIZE:
        print(f"Warning: Dataset is smaller than {SAMPLE_SIZE}. Using all {len(full_data)} rows.")
        SAMPLE_SIZE = len(full_data)

    # Create a random sample
    print(f"Creating a random sample of {SAMPLE_SIZE} rows...")
    sample_data = full_data.sample(n=SAMPLE_SIZE, random_state=42)

    # Save it to the 'data' folder
    sample_data.to_csv(SAMPLE_DATA_PATH, index=False)

    print(f"✅ Success! Sample file saved to {SAMPLE_DATA_PATH}")
    print(f"The average lap time is: {sample_data['Lap_Time_Seconds'].mean()}")