from flask import Flask, request, render_template
import requests

app = Flask(__name__)
MODEL_URL = "http://sentiment-model.default.127.0.0.1.nip.io/v1/models/sentiment:predict"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        review = request.form.get("review", "")
        if not review:
            return render_template("index.html", error="Please enter a review")
        try:
            payload = {"instances": [{"text": review}]}
            response = requests.post(MODEL_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()["predictions"][0]
            sentiment = "Positive" if result["positive"] > result["negative"] else "Negative"
            return render_template(
                "index.html",
                review=review,
                sentiment=sentiment,
                positive=result["positive"],
                negative=result["negative"]
            )
        except requests.RequestException as e:
            return render_template("index.html", error=f"Prediction failed: {str(e)}")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)