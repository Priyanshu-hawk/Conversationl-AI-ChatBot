# langchain chatbot
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_groq import ChatGroq
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# system imports
from uuid import uuid4
import json
import os
from threading import Thread
from datetime import datetime
import requests

# config
from config import bot_config

# env
from dotenv import load_dotenv
load_dotenv()

#tools to be used deterministic
class ToolParameter(BaseModel):
    name: str
    description: str
    required: bool = False
    default: Optional[Any] = None

class Tool(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    examples: List[str]

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

# using local storage for now, will move to NoSQL later or Vector DB
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
        self.user_context = self.parse_user_personal_info(self.user_id)
        if self.user_context:
            self.system_prompt += f"\n\nUser Information:\n{self.user_context}"

        self.available_tools = {
            "weather": self.get_weather,
            "time": self.get_current_time,
            # "news": self.get_news # will add later
        }

        self.tools_registry = {
        "weather": Tool(
            name="weather",
            description="Get current weather information for a location",
            parameters=[
                ToolParameter(
                    name="city",
                    description="The city name to get weather for",
                    required=True
                )
            ],
            examples=[
                "What's the weather in London?",
                "How's the weather today in New York?",
                "Tell me the weather in Tokyo"
            ]
        ),
        "time": Tool(
            name="time", 
            description="Get the current time",
            parameters=[],
            examples=[
                "What time is it?",
                "Tell me the current time",
                "What's the time now?"
            ]
        )
    }

    def greet(self, user_id: str) -> str:
        print(user_id)
        print(user_id in self.storage.get_all_users())
        if user_id not in self.storage.get_all_users():
            return "Hello! How can I help you today?"
        
        name = self.storage.load_personal_info(user_id, "name")
        if name:
            return f"Hello {name[0]}! How can I help you today?"
        
    # Functions to handel Weather inquiries and Time inquiries
    def _check_tool_query(self, user_input: str) -> dict:
        tools_context = "\n\n".join([
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Parameters: {[p.name for p in tool.parameters]}\n"
            f"Example queries:\n" + "\n".join([f"- {ex}" for ex in tool.examples])
            for tool in self.tools_registry.values()
        ])

        system_prompt = f"""
You are a tool parser that analyzes user input to detect tool requests.

User Context:
{self.user_context}

Available tools and their specifications:
{tools_context}

For each user message:
1. Determine if they're requesting to use any of the available tools
2. Extract relevant parameters based on the tool's parameter specifications
3. If parameters are not explicitly mentioned, try to infer them from user context
4. Validate that required parameters are present

Consider user preferences and location when resolving parameters.
For example, if user asks about weather without specifying city, use their stored location.

Output format:
{{
    "tool": "<tool_name or 'none'>",
    "params": {{
        "param1": "value1",
        ...
    }},
    "confidence": <float between 0 and 1>,
    "context_used": <boolean>
}}

Return 'none' if no tool is requested or if required parameters are missing.
    """
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ])
            
            result = json.loads(response.content)
            
            # Validate parameters if tool is detected
            if result["tool"] != "none":
                tool = self.tools_registry[result["tool"]]
                required_params = {p.name for p in tool.parameters if p.required}
                provided_params = set(result["params"].keys())
                
                # Try to fill missing parameters from user context
                if not required_params.issubset(provided_params):
                    if "city" in required_params and "location" in self.user_context:
                        result["params"]["city"] = self.storage.load_personal_info(self.user_id, "location")[0]
                        result["context_used"] = True
                
                # Final validation
                if not required_params.issubset(set(result["params"].keys())):
                    missing = required_params - set(result["params"].keys())
                    print(f"Missing required parameters: {missing}")
                    return {"tool": "none", "params": {}, "confidence": 0, "context_used": False}
                    
            return result
        except Exception as e:
            print(f"Tool parsing error: {str(e)}")
            return {"tool": "none", "params": {}, "confidence": 0, "context_used": False}
        
    def execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute tool with parameter validation"""
        try:
            if tool_name not in self.available_tools:
                return f"Tool '{tool_name}' not found."
            
            tool = self.tools_registry[tool_name]
            valid_params = {}
            
            for param in tool.parameters:
                if param.name in params:
                    valid_params[param.name] = params[param.name]
                elif param.default is not None:
                    valid_params[param.name] = param.default
                    
            tool_func = self.available_tools[tool_name]
            result = tool_func(**valid_params)
            return result
        except Exception as e:
            print(f"Tool execution error: {str(e)}")
            return f"Sorry, I encountered an error while using the {tool_name} tool."

    def get_weather(self, city: str = None) -> str:
        try:
            if not city:
                # naive implementation, Will improve later with vector embeddings
                city = self.storage.load_personal_info(self.user_id, "location")

            print(f"Fetching weather for {city}")
            

            api_key = os.getenv("WEATHER_API_KEY")
            url = "http://api.weatherapi.com/v1"
            complete_url = f"{url}/current.json?key={api_key}&q={city}"
            response = requests.get(complete_url)
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            temp_c = data['current']['temp_c']
            time = "night" if data['current']['is_day'] == 0 else "day"
            condition = data['current']['condition']['text']
            
            return f"The temperature in {city} is {temp_c}°C. It is {time} time and the weather condition is {condition}."
        except requests.RequestException as e:
            print(f"Weather API error: {str(e)}")
            return f"Sorry, I couldn't fetch the weather for {city}."

    def get_current_time(self) -> str:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return f"The current time is {current_time}."
        
    
    #Post Procesing of the response - TODO in thread
    def _check_and_add_personal_info(self, human_message: str, response: str, user_id: str) -> bool:
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

Do not provide any additional commentary or explanation. Be precise and structured. Only return the detected information.
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
            print(f"Category: {category}, Detail: {detail}")
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
            self.user_context
            tool_result = self._check_tool_query(user_input)
            
            if tool_result["tool"] != "none":
                return self.execute_tool(tool_result["tool"], tool_result["params"])

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
    