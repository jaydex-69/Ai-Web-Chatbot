from flask import Flask, jsonify, request, render_template
from pathlib import Path
from utils import predict_class, get_response

folder = Path(__file__).parent/'templates'

app = Flask(__name__, template_folder=folder)

app.route('/')
def index():
    return render_template('index.html')

app.route('handle_message', method=['POST'])
def handle_message():
    message = request.json['message']
    prediction = predict_class(message)
    response = get_response(prediction)

    return jsonify({'response': response})

app.run('0.0.0.0', debug=True)