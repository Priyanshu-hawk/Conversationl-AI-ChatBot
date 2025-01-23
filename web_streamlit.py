import streamlit as st
from streamlit_geolocation import streamlit_geolocation

from chatbot import Chatbot, ChatBotLangchain


# Page config
st.set_page_config(page_title="Priyanshu's AI Chatbot", page_icon="🤖")
st.title("AI Chatbot")

# Initialize chatbot
# chatbot = Chatbot()
chatbot = ChatBotLangchain()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # # check implem.
    # if prompt == "location":
    #     location = streamlit_geolocation()
    #     st.write(location)
    
    # Get bot response
    response = chatbot.get_response(prompt)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Add some styling
st.markdown("""
    <style>
    .stChat {
        padding: 20px;
    }
    .stChatMessage {
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)