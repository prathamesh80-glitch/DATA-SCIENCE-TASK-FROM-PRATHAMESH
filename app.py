from flask import Flask, request, jsonify
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    area = data['Area']
    bedrooms = data['Bedrooms']

    prediction = model.predict([[area, bedrooms]])

    return jsonify({'Predicted Price': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
