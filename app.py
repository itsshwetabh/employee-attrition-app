from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_input = [
        data['satisfaction_level'],
        data['last_evaluation'],
        data['number_project'],
        data['average_montly_hours'],
        data['time_spend_company'],
        data['Work_accident'],
        data['promotion_last_5years'],
        data['low'],
        data['medium']
    ]
    
    model = joblib.load('model.pkl')
    prediction = model.predict([model_input])[0]
    
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
