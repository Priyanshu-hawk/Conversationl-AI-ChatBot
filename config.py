bot_config = {
    "model": {
        "core_llm": {"name": "llama-3.3-70b-versatile",
                     "max_tokens": 1024,
                     "temperature": 0.2}, # Randomness of the for versatile responses
        "personal_info": {"name": "llama-3.3-70b-versatile",
                            "max_tokens": 512,
                            "temperature": 0.0}, # Deterministic responses for stict cases 
    },
    "conversation" : {
        "max_history": 120,
    },
    "bot": {
        "name": "Priyanshu's Bot",
        "role": "assistant",
        "prompt": {
            "system": "You are a helpful assistant. You name is Personal Chatbot. You reply with concise answers. Answer all questions to the best of your ability. If you don't know the answer to a question, you can say 'unknown'",
        },
        "system_error": "I'm sorry, I encountered an error. Could you please try again?",
        "unknown": "I'm sorry, I don't have an answer to that question.",
    }
}