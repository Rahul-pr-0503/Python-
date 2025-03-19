from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("fertilizer_model.pkl")

@app.route("/")
def home():
    return "Fertilizer Recommendation API is Running!"

@app.route("/recommend", methods=["POST"])
def recommend_fertilizer():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid input"}), 400

        soil_data = np.array([[data["pH"], data["N"], data["P"], data["K"], data["Moisture"]]])
        recommendation = model.predict(soil_data)[0]
        return jsonify({"recommended_fertilizer": recommendation})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
