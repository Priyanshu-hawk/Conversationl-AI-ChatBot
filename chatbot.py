from dotenv import load_dotenv
import os

# basic chatbot
from groq import Groq

# langchain chatbot
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from uuid import uuid4

# config
from config import bot_config
load_dotenv()

# Basic chatbot
class Chatbot:
    def __init__(self):
        self.llm = Groq() # api key is in env 
        self.conversation_history = [{
            "role": "system",
            "content": bot_config["bot"]["prompt"]["system"]
        }]
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
        
class ChatBotLangchain:
    def __init__(self):
        self.llm = ChatGroq(model=bot_config["model"]["name"]) # api key is in env
        
        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self._call_model)
        self.workflow.add_edge(START, "model")
        
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        self.system_prompt = (bot_config["bot"]["prompt"]["system"])

    def _call_model(self, state: MessagesState):
        system_message = SystemMessage(content=self.system_prompt)
        message_history = state["messages"][:-1]
        
        if len(message_history) >= 120:
            last_human_message = state["messages"][-1]
            summary_prompt = "Distill the above chat messages into a single summary message. Include as many specific details as you can."
            summary_message = self.llm.invoke(message_history + [HumanMessage(content=summary_prompt)])
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
            human_message = HumanMessage(content=last_human_message.content)
            response = self.llm.invoke([system_message, summary_message, human_message])
            message_updates = [summary_message, human_message, response] + delete_messages
        else:
            message_updates = self.llm.invoke([system_message] + state["messages"])
        
        return {"messages": message_updates}

    def get_response(self, user_input: str) -> str:
        try:
            state = {"messages": [HumanMessage(content=user_input)]}
            response = self.app.invoke(state, config={"configurable": {"thread_id": str(uuid4())}})
            
            bot_response = response["messages"][-1].content
            if not bot_response or bot_response.lower() == "unknown":
                return "I'm sorry, I don't have an answer to that question."
            
            return bot_response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return "I'm sorry, I encountered an error. Could you please try again?"