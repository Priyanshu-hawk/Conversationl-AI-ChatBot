from flask import Flask, render_template, request, jsonify
from chatbot import Chatbot
import os

app = Flask(__name__)
chatbot = Chatbot()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message', '')
    if not message:
        return jsonify({'error': 'Empty message'}), 400
    
    response = chatbot.get_response(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)