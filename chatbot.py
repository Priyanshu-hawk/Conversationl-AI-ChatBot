# langchain chatbot
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_groq import ChatGroq

# system imports
from uuid import uuid4
import json
import os
from threading import Thread

# config
from config import bot_config

# env
from dotenv import load_dotenv
load_dotenv()

#Personal Info List
personal_info_list = [
            "name", "gender", "age", "nationality", "location", "email", "phone_number", 
            "social_media", "language_preference", "hobbies", "food_preferences",
            "movies", "books", "music", "profession", "company", "education",
            "skills", "family", "marital_status", "friends", "medical_conditions",
            "allergies", "fitness_goals", "income", "banking_preferences", "investments",
            "travel_locations", "favorite_destinations", "mode_of_transport",
            "devices", "apps", "personal_goals", "career_goals"
        ]

class LocalJSONStorage:
    def __init__(self, file_path="personal_info.json"):
        self.file_path = file_path
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                json.dump({}, file)

    def save_personal_info(self, user_id: str, info_type: str, info: str):
        # Load existing data
        with open(self.file_path, "r") as file:
            data = json.load(file)
        
        # update the data
        if user_id not in data:
            data[user_id] = {}
        if info_type not in data[user_id]:
            data[user_id][info_type] = []
        
        # no duplicates
        if info not in data[user_id][info_type]:
            data[user_id][info_type].append(info)
    
        with open(self.file_path, "w") as file:
            json.dump(data, file, indent=4)


    def load_personal_info(self, user_id: str, info_type: str = None):
        # Load existing data
        with open(self.file_path, "r") as file:
            data = json.load(file)
        
        if info_type == "all":
            return data.get(user_id, {})

        # Retrieve the user's data
        if user_id not in data:
            return None
        if info_type:
            return data[user_id].get(info_type, [])
        return data[user_id]
    
    def get_all_users(self):
        with open(self.file_path, "r") as file:
            data = json.load(file)
        return list(data.keys())
        
class ChatBotLangchain:
    def __init__(self, user_id: str):
        self.llm = ChatGroq(model=bot_config["model"]["core_llm"]["name"]) # api key is in env
        self.llm_personal_info = ChatGroq(model=bot_config["model"]["personal_info"]["name"]) # api key is in env
        
        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self._call_model)
        self.workflow.add_edge(START, "model")
        
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        self.thread_id = str(uuid4())
        self.user_id = user_id
        self.storage = LocalJSONStorage()

        self.system_prompt = (bot_config["bot"]["prompt"]["system"])
        user_context = self.parse_user_personal_info(self.user_id)
        if user_context:
            self.system_prompt += f"\n\nUser Information:\n{user_context}"

    def greet(self, user_id: str) -> str:
        print(user_id)
        print(user_id in self.storage.get_all_users())
        if user_id not in self.storage.get_all_users():
            return "Hello! How can I help you today?"
        
        name = self.storage.load_personal_info(user_id, "name")
        if name:
            return f"Hello {name[0]}! How can I help you today?"
    
    #Post Procesing of the response - TODO in thread
    def _check_and_add_personal_info(self, human_message: str, response: str, user_id: str) -> bool:
        """
        Checks if a chat contains personal information and saves it if present.

        Args:
            human_message (str): The human's input message.
            response (str): The AI's response message.
            user_id (str): The user ID for saving personal information.

        Returns:
            bool: True if personal information was found and saved, False otherwise.
        """
        # System prompt for the LLM
        system_prompt = """
Your job is to analyze a brief conversation and extract any details about a person's personal life, 
categorized as follows:
{}

Follow these steps:
1. Check the conversation for any details matching the above categories.
2. If no details are found, respond with "NO PERSONAL INFORMATION".
3. If personal information is found, return each identified piece in the format:
"<Category>: <Detail>".

Do not provide any additional commentary or explanation. Be precise and structured.
""".format("\n".join(personal_info_list))

        # Combine chat history
        recent_chat = f"""
Human: {human_message}
AI: {response}
"""
        detected_info = self.llm_personal_info.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=recent_chat)]
        ).content.strip()

        print("Detected Info:", detected_info)

        # Check if any personal information was detected
        if detected_info.upper() == "NO PERSONAL INFORMATION":
            return False

        # Parse and save personal information
        personal_info_entries = self._parse_detected_info(detected_info)
        for category, detail in personal_info_entries:
            # print(f"Category: {category}, Detail: {detail}")
            self.save_personal_info(user_id, category, detail)

        return True

    def _parse_detected_info(self, detected_info: str) -> list:
        entries = []
        for line in detected_info.split("\n"):
            if ": " in line:
                category, detail = line.split(": ", 1)
                entries.append((category.strip(), detail.strip()))
        return entries
    
    def save_personal_info(self, user_id: str, info_type: str, info: str):
        if info_type not in personal_info_list:
            raise ValueError(f"Invalid info_type: {info_type}")
        self.storage.save_personal_info(user_id, info_type, info)
    

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
    
    def parse_user_personal_info(self, user_id: str) -> str:
        try:
            # Get all stored info for user
            user_data = self.storage.load_personal_info(user_id, "all")
            if not user_data:
                return ""

            # Group information by categories
            categories = {
                "Basic Info": ["name", "age", "gender", "nationality", "location"],
                "Contact": ["email", "phone_number", "social_media"],
                "Preferences": ["language_preference", "food_preferences", "movies", "books", "music"],
                "Professional": ["profession", "company", "education", "skills"],
                "Personal": ["hobbies", "family", "marital_status"],
                "Health": ["medical_conditions", "allergies", "fitness_goals"],
                "Other": []
            }

            formatted_info = []
            
            # Process each category
            for category, fields in categories.items():
                category_info = []
                for field in fields:
                    if field in user_data and user_data[field]:
                        values = user_data[field]
                        if isinstance(values, list):
                            value_str = ", ".join(values)
                        else:
                            value_str = str(values)
                        category_info.append(f"{field}: {value_str}")
                
                # Add non-empty categories
                if category_info:
                    formatted_info.append(f"{category}:\n" + "\n".join(category_info))

            # Add any uncategorized information
            other_info = []
            for field, values in user_data.items():
                if not any(field in fields for fields in categories.values()):
                    if isinstance(values, list):
                        value_str = ", ".join(values)
                    else:
                        value_str = str(values)
                    other_info.append(f"{field}: {value_str}")
            
            if other_info:
                formatted_info.append("Other Information:\n" + "\n".join(other_info))

            print("Formatted Info:", formatted_info)

            return "\n\n".join(formatted_info)

        except Exception as e:
            print(f"Error parsing personal info: {str(e)}")
            return ""

    def get_response(self, user_input: str, user_id: str) -> str:
        try:
            state = {"messages": [HumanMessage(content=user_input)]}
            response = self.app.invoke(state, config={"configurable": {"thread_id": self.thread_id}})
            
            bot_response = response["messages"][-1].content
            if not bot_response or bot_response.lower() == "unknown":
                return "I'm sorry, I don't have an answer to that question."
            
            # Check for personal information in background and save if found
            thread = Thread(target=self._check_and_add_personal_info, args=(user_input, bot_response, user_id))
            thread.start()
    
            return bot_response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return "I'm sorry, I encountered an error. Could you please try again?"
        

#check personal
if __name__ == "__main__":
    chatbot = ChatBotLangchain("test_user")
    user_input = "What is your name?"
    response = chatbot.get_response(user_input, "test_user")
    print(response)
    user_input = "I am from India."
    response = chatbot.get_response(user_input, "test_user")
    print(response)
    