import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dataset (replace with actual data)
data = pd.DataFrame({
    "pH": [6.5, 5.8, 7.0, 6.2, 6.8],
    "N": [30, 40, 35, 45, 50],
    "P": [40, 50, 45, 55, 60],
    "K": [50, 60, 55, 65, 70],
    "Moisture": [20, 25, 22, 27, 30],
    "Fertilizer": ["Urea", "DAP", "NPK", "Organic", "Compost"]
})

# Split dataset
X = data.drop(columns=["Fertilizer"])
y = data["Fertilizer"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fertilizer_model.pkl")

print("Model trained and saved as fertilizer_model.pkl")
