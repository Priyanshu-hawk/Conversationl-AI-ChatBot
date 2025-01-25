from flask import Flask, render_template, request, jsonify
from chatbot import ChatBotLangchain
import os
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
# chatbot = Chatbot()
# chatbot = ChatBotLangchain()

session_state = {}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message', '')
    user_id = request.json.get('user_id', '')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    if user_id in session_state:
        chatbot = session_state[user_id]
    else:
        chatbot = ChatBotLangchain(user_id)
        session_state[user_id] = chatbot
    if not message:
        return jsonify({'error': 'Empty message'}), 400
    
    response = chatbot.get_response(message, user_id)
    return jsonify({'response': response})

@app.route('/greet', methods=['POST'])
def greet():
    try:
        user_id = request.json.get('user_id', '')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400

        if user_id in session_state:
            chatbot = session_state[user_id]
        else:
            chatbot = ChatBotLangchain(user_id)
            session_state[user_id] = chatbot
        response = chatbot.greet(user_id)

        print(response)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = 7331     # NICE
    try:
        if os.getenv("ENVIORNMENT") == "dev":
            app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)

        if os.getenv("ENVIORNMENT") == "prod":
            http_server = WSGIServer(('', port), app)
            http_server.serve_forever()
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        print("Error", e)
    finally:
        print("oops! finally block")
        print("Server Stopped")
        pass