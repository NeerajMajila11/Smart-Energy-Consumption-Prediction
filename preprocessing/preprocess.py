import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/energydata_complete.csv")

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

# -----------------------------
# Convert Date Column
# -----------------------------
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")

print("\nDate column converted successfully!")

# -----------------------------
# Dataset Information
# -----------------------------
print("\nDataset Info")
print(df.info())

# -----------------------------
# Check Missing Values
# -----------------------------
print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------
# Check Duplicate Rows
# -----------------------------
print("\nDuplicate Rows:", df.duplicated().sum())

# -----------------------------
# Statistical Summary
# -----------------------------
print("\nStatistical Summary")
print(df.describe())

# -----------------------------
# Save Clean Dataset
# -----------------------------
df.to_csv("dataset/energy_clean.csv", index=False)

print("\nClean dataset saved as dataset/energy_clean.csv")