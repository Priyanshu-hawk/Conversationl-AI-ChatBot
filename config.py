bot_config = {
    "model": {
        "name": "llama-3.3-70b-versatile", 
        "max_tokens": 1024,
        "temperature": 0.1, # defines the randomness of the output 
    },
    "conversation" : {
        "max_history": 120,
    },
    "bot": {
        "name": "Priyanshu's Bot",
        "role": "assistant",
        "prompt": {
            "system": "You are a helpful assistant. You name is Priyanshu's Bot. You reply with concise answers. Answer all questions to the best of your ability. If you don't know the answer to a question, you can say 'unknown'",
        },
        "system_error": "I'm sorry, I encountered an error. Could you please try again?",
        "unknown": "I'm sorry, I don't have an answer to that question.",
    }
}