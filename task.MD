# Assignment Title: Build a Simple Conversational AI Chatbot

## Objective
To evaluate the candidate's ability to:
1. Understand the basics of conversational AI and chatbot development.  
2. Use pre-trained models to build a functional chatbot.  
3. Demonstrate creativity and problem-solving skills in creating a simple, user-friendly chatbot interface.

## Task Description
Create a Conversational AI Chatbot that can answer basic questions or have simple conversations with a user. The chatbot should have the following features:

1. **Input**  
   - Accept text input from the user (e.g., via a command-line interface, a simple web app, or a chat interface).

2. **Processing**  
   - Use a pre-trained conversational AI model (e.g., OpenAI’s GPT-3.5, Hugging Face models, or similar).  
   - Ensure the chatbot responds coherently to user queries.

3. **Output**  
   - Display the chatbot’s response in real-time.

## Functional Requirements
1. The chatbot should be able to:  
   - Greet the user (e.g., “Hello! How can I help you today?”).  
   - Respond to simple general knowledge questions (e.g., “What’s the capital of France?”).  
   - Handle casual conversation (e.g., “How’s the weather today?” or “Tell me a joke.”)

2. The chatbot should handle basic edge cases:  
   - If it doesn’t understand a query, respond with something like:  
     “I’m sorry, I didn’t understand that. Could you please rephrase?”

## Bonus Features (Optional)
1. **Predefined Intent Recognition**  
   - Recognize specific intents like:  
     - Weather inquiries: “What’s the weather like today?”  
     - Time-related queries: “What time is it?”  
   - Respond accordingly.

2. **Web Interface**  
   - Create a simple web-based interface using a framework like Flask, Streamlit, or React.js.

3. **Personalization**  
   - Allow the user to provide their name and personalize responses (e.g., “Hi John, how can I assist you?”).

4. **Memory**  
   - Enable the chatbot to remember the context of the conversation within a session (e.g., “Can you tell me more about that?”).

## Technical Requirements
1. **Programming Language**  
   - Use Python for development.

2. **Pre-Trained Model**  
   - Use a conversational AI model such as:  
     - OpenAI's API (GPT-3.5 or similar).  
     - Hugging Face’s transformers library with models like DialoGPT or T5.  
     - Or any basic chatbot library like ChatterBot.

3. **Libraries/Tools**  
   - Suggested libraries: transformers, flask (for a web interface), nltk, or any other you’re comfortable with.

4. **Error Handling**  
   - Handle empty or invalid input gracefully.

5. **Documentation**  
   - Provide clear instructions in a README.md file on how to set up and run the chatbot.

## Deliverables
1. **Source Code**  
   - A GitHub repository containing:  
     - The Python code for the chatbot.  
     - Any configuration or requirement files (e.g., requirements.txt).

2. **README.md**  
   - Clear setup instructions.  
   - Examples of queries the chatbot can handle.

3. **Sample Output**  
   - Provide a few examples of conversations with the chatbot.

4. **Optional Deliverables**  
   - A link to the web-based interface (if implemented).

## Evaluation Criteria
1. **Understanding of Conversational AI**  
   - Does the chatbot handle user queries logically and conversationally?

2. **Functionality**  
   - Does the chatbot work as described in the task requirements?

3. **Code Quality**  
   - Is the code clean, modular, and readable?  
   - Are best practices followed for error handling and maintainability?

4. **Creativity**  
   - Have any bonus features been implemented, and are they functional?

5. **Documentation**  
   - Is the README.md file clear and helpful?

## Tips for Candidates
1. **Start Simple**  
   - Focus on getting the chatbot to respond to a few basic queries before adding extra features.

2. **Use Online Resources**  
   - Look for tutorials or documentation for tools like Hugging Face or OpenAI’s API.

3. **Test as You Go**  
   - Regularly test the chatbot to ensure it behaves as expected.

4. **Add Comments**  
   - Write clear comments in your code to explain key sections.