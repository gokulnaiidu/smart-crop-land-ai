from flask import Flask, request, jsonify
import joblib

# ==================================================
# SMART CROP LAND AI SERVER
# ==================================================

app = Flask(__name__)

# ==================================================
# LOAD AI MODEL
# ==================================================

try:
    model = joblib.load("crop_model.pkl")
    print("AI model loaded successfully!")

except Exception as e:
    print("ERROR loading AI model:")
    print(e)
    model = None


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/", methods=["GET"])
def home():

    return "Smart Crop Land AI Server is Running!"


# ==================================================
# AI PREDICTION
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON data from ESP32
        data = request.get_json()

        print()
        print("==============================")
        print("NEW AI REQUEST")
        print("==============================")

        print("Received data:")
        print(data)

        # ------------------------------------------------
        # Read sensor values
        # ------------------------------------------------

        ph = float(data["ph"])
        moisture = float(data["moisture"])
        temperature = float(data["temperature"])

        print("pH:", ph)
        print("Moisture:", moisture)
        print("Temperature:", temperature)

        # ------------------------------------------------
        # AI prediction
        # ------------------------------------------------

        prediction = model.predict([
            [ph, moisture, temperature]
        ])

        crop = prediction[0]

        # ------------------------------------------------
        # Confidence
        # ------------------------------------------------

        probabilities = model.predict_proba([
            [ph, moisture, temperature]
        ])

        confidence = max(probabilities[0]) * 100

        print()
        print("AI RESULT")
        print("Crop:", crop)
        print("Confidence:", round(confidence, 2), "%")
        print("==============================")

        # ------------------------------------------------
        # Send response to ESP32
        # ------------------------------------------------

        return jsonify({
            "crop": str(crop),
            "confidence": round(confidence, 2),
            "ph": ph,
            "moisture": moisture,
            "temperature": temperature
        })

    except Exception as e:

        print()
        print("ERROR:")
        print(e)

        return jsonify({
            "error": str(e)
        }), 400


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":

    print()
    print("==============================")
    print("SMART CROP LAND AI SERVER")
    print("==============================")
    print("Server starting...")
    print()
    print("Laptop IP:")
    print("10.46.55.178")
    print()
    print("AI Server:")
    print("http://10.46.55.178:5000")
    print()
    print("Prediction URL:")
    print("http://10.46.55.178:5000/predict")
    print("==============================")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )