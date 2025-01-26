# AI Chatbot with Personalization and Web Interface

A conversational AI chatbot built with Flask and LangChain that supports personalization, weather queries, time queries, and persistent memory.

## Features

- Conversational AI powered by Groq's LLama 3.3 70B model
- User personalization and context memory
- Real-time weather information
- Current time queries 
- Custom web interface

## Bonus Features

- Persistent storage of user information (Beta)

## Tech Stack

- **Backend**: Python 3.10+ with Flask
- **Frontend**: HTML5, CSS, JavaScript
- **AI Model**: Groq LLama 3.3 70B
- **Storage**: JSON-based local storage
- **APIs**: Weather API integration

## Prerequisites

- Python 3.10 or higher
- Groq API key
- Weather API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
- For Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```
- For Windows (May this works, I don't have windows machine to test):
```bash
python -m venv venv
venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env

 file from template:
```bash
cp .env.example .env
```

5. Configure environment variables in 

- Groq API key - https://console.groq.com/keys
- Weather API key - https://www.weatherapi.com/my/

File: `.env`
```
GROQ_API_KEY=<your-groq-api-key>
WEATHER_API_KEY=<your-weather-api-key>
ENVIRONMENT=dev
```

## Running the Application

### Local Development
```bash
python web.py
```
The application will be available at [http://localhost:5000](http://localhost:5000)

### Production Deployment
```bash
gunicorn --bind :$PORT --workers=1 web:app
```

## Usage

1. Open the web interface in your browser
2. Enter a user ID to start chatting. (The chatbot will remember your information)
3. Type messages in the chat input field

### Example Queries

- Basic conversation:
  - "Hello, how are you?"

- Personal information:
  - "My name is Prip"
  - "I live in Moscow"

- Weather queries:
  - "What's the weather like in Raipur?"
  - "How's the weather today?" (uses saved location)

- Time queries:
  - "What time is it?"
  - "Tell me the current time"

- Context memory:
  - "Can you tell me more about that?"
  - "Remind me of my name"

### ⭐ Example Conversations with the chatbot: [Conversation examples here](conversation_examples.md)

## Project Structure
Commmand to generate tree on linux - `tree Conversationl-AI-ChatBot-Vizares -L 2`

```
Conversationl-AI-ChatBot
├── chatbot.py
├── config.py
├── personal_info.json
├── Procfile
├── __pycache__
│   <cache-files>
├── Readme.MD
├── requirements.txt
├── static
│   └── css
├── task.MD
├── templates
│   └── index.html
├── venv
│   <venv-files>
└── web.py
```

## Deployment ⚠️⚠️⚠️
### The reason i put this in the deployment section at the end is because this project is not properly working on the deployment server. 
This project is deployed on Koyeb. But Few things are not working as expected. Because of the limitations of the free tier of Koyeb. This things are not working are:
- Persistent storage of user information (Some times it works, sometimes it doesn't)
- Time queries (It give the time of the server where the code is running)
- Slow API call for weather API.

#### Deployment Link: [ChatBot](https://disciplinary-odelia-priyanshu-hawk-a555e99d.koyeb.app/)

## Configuration

Modify : config.py
    
to adjust:
- Model parameters
- Conversation settings
- System prompts
- Error messages