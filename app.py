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

        # ==================================================
        # GET JSON DATA FROM ESP32
        # ==================================================

        data = request.get_json()

        print()
        print("==============================")
        print("NEW AI REQUEST")
        print("==============================")

        print("Received data:")
        print(data)

        # ==================================================
        # READ SENSOR VALUES
        # ==================================================

        ph = float(data["ph"])
        moisture = float(data["moisture"])
        temperature = float(data["temperature"])

        print("pH:", ph)
        print("Moisture:", moisture)
        print("Temperature:", temperature)

        # ==================================================
        # STRICT MOISTURE RANGE
        # ==================================================

        if 2450 <= moisture <= 2650:

            crop = "Chili"

        elif 2250 <= moisture <= 2449:

            crop = "Corn"

        elif 2050 <= moisture <= 2249:

            crop = "Tomato"

        elif 1750 <= moisture <= 2049:

            crop = "Peanut"

        else:

            crop = "Unknown"


        # ==================================================
        # CONFIDENCE
        # ==================================================

        if crop != "Unknown":

            if model is not None:

                try:

                    probabilities = model.predict_proba([
                        [ph, moisture, temperature]
                    ])

                    confidence = max(
                        probabilities[0]
                    ) * 100

                except Exception:

                    confidence = 100.0

            else:

                confidence = 100.0

        else:

            confidence = 0.0


        # ==================================================
        # PRINT RESULT
        # ==================================================

        print()
        print("==============================")
        print("SMART CROP LAND RESULT")
        print("==============================")

        print("Moisture:", moisture)
        print("Crop:", crop)
        print(
            "Confidence:",
            round(confidence, 2),
            "%"
        )

        print("==============================")


        # ==================================================
        # SEND RESPONSE TO ESP32
        # ==================================================

        return jsonify({

            "crop": crop,

            "confidence": round(
                confidence,
                2
            ),

            "ph": ph,

            "moisture": moisture,

            "temperature": temperature

        })


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as e:

        print()
        print("==============================")
        print("ERROR")
        print("==============================")

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

    print("AI Server:")
    print(
        "http://10.46.55.178:5000"
    )

    print()

    print("Prediction URL:")
    print(
        "http://10.46.55.178:5000/predict"
    )

    print("==============================")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
