from flask import Flask, jsonify, request
from utils import predict_class, get_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # IMPORTANT for React

@app.route('/chat', methods=['POST'])
def handle_message():
    data = request.json
    message = data.get('message')

    prediction = predict_class(message)
    response = get_response(prediction)

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)