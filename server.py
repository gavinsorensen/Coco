from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello, world!"

@app.route("url = "http://172.31.8.24/api/v1/predict"", methods=["POST"])
def predict():
    data = request.json
    api_key = data.get('api_key')
    user_input = data.get('user_input')

    if api_key and user_input:
        url = "http://3.237.78.240/api/v1/predict"  # Replace with your deployed API URL
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": api_key,
            "user_input": user_input
        }
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json().get("result")
            return jsonify({"result": result})
        else:
            error_message = "An error occurred while processing the request."
            return jsonify({"error": error_message})

    return jsonify({"error": "Invalid API key or user input"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
