# app.py
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('sentiment_model.pkl')

@app.route('/')
def home():
    # This renders the homepage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # This function gets called when the user submits text
    data = request.get_json()
    text = data['text']
    
    # Use the model to predict
    prediction = model.predict([text])[0]
    
    # You can also get the probability if you want to show confidence
    # probabilities = model.predict_proba([text])[0]
    
    # Send the result back to the webpage
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)