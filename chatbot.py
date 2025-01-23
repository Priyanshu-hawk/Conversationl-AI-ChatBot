from dotenv import load_dotenv
import os
from groq import Groq

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq


# config
from config import bot_config

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
system_prompt = {
    "role": "system",
    "content": "You are a helpful assistant You name is Priyanshu's Bot. You reply with short and concise answers. If you don't know the answer to a question, you can say 'unknown'"
}


# Basic chatbot
class Chatbot:
    def __init__(self):
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.conversation_history = [system_prompt]
    def get_response(self, user_input):
        try:
            self.conversation_history.append({"role": "user", "content": user_input})
            
            response = self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=self.conversation_history,
            )
            
            bot_response = response.choices[0].message.content
            if bot_response == "":
                bot_response = "I'm sorry, I don't have an answer to that question."
            
            if bot_response.lower() == "unknown":
                bot_response = "I'm sorry, I don't have an answer to that question."
            
            else:
                self.conversation_history.append({"role": "assistant", "content": bot_response})
            
            return bot_response
        except Exception as e:
            return "I'm sorry, I encountered an error. Could you please try again?"
        
class chatbotLangchain:
    def __init__(self):
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='llama-3.3-70b-versatile')
        self.memory = ConversationBufferWindowMemory(k=bot_config["conversation"]["max_history"], memory_key="chat_history", return_messages=True)

    def get_response(self, user_input):
        if user_input == "":
            return "Please enter a valid input."
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=bot_config["bot"]["prompt"]["system"]
                    ),
                    MessagesPlaceholder(
                        variable_name="chat_history"
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "{human_input}"
                    ),
                ]
            )

            conversation = prompt | self.llm

            response = conversation(
                chat_history=self.memory,
                human_input=user_input
            )
            return response
        except Exception as e:
            print(e)
            return bot_config["bot"]["system_error"]