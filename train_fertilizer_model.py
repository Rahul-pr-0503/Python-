
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("dataset.csv")

# Ensure we have diverse fertilizers
print("Unique Fertilizers:", df["Fertilizer"].unique())

# Balance the dataset (optional)
df_balanced = df.groupby("Fertilizer").apply(lambda x: x.sample(100, replace=True)).reset_index(drop=True)

# Prepare features and labels
X = df_balanced[["pH", "N", "P", "K", "Moisture"]]
y = df_balanced["Fertilizer"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "fertilizer_model.pkl")
print("Model trained and saved successfully!")
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("dataset.csv")

# Ensure we have diverse fertilizers
print("Unique Fertilizers:", df["Fertilizer"].unique())

# Balance the dataset (optional)
df_balanced = df.groupby("Fertilizer").apply(lambda x: x.sample(100, replace=True)).reset_index(drop=True)

# Prepare features and labels
X = df_balanced[["pH", "N", "P", "K", "Moisture"]]
y = df_balanced["Fertilizer"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "fertilizer_model.pkl")
print("Model trained and saved successfully!")
