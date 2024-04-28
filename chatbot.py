import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from settings import MAX_BOT_MEMORY_SIZE, LLM_MOD
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from database import Session
from models import ChatHistory
import os
import json  # Import json to handle serialization of list

st.title("Chat with Groq!")
st.write("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

def load_chat_history():
    with Session() as session:
        last_entry = session.query(ChatHistory).order_by(ChatHistory.id.desc()).first()
        if last_entry:
            try:
                loaded_data = json.loads(last_entry.chat_history)
                # Validate data format or transform if necessary
                if all("human" in message and "AI" in message for message in loaded_data):
                    return loaded_data
                else:
                    print("Loaded data does not match expected format.")
                    return []
            except json.JSONDecodeError as e:
                print("Failed to decode JSON:", str(e), "Data:", last_entry.chat_history)
                return []
        return []

# Initialize or load chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Create a memory buffer for the chatbot
memory = ConversationBufferWindowMemory(k=MAX_BOT_MEMORY_SIZE)

# Populate memory with the loaded chat history
for message in st.session_state.chat_history:
    if "human" in message and "AI" in message:
        memory.save_context({"input": message["human"]}, {"output": message["AI"]})
    else:
        print("Message format error:", message)

groq_chat = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=LLM_MOD, temperature=0.5)
conversation = ConversationChain(llm=groq_chat, memory=memory)

with st.expander("Chat"):
    user_question = st.text_area("Ask a question:", height=200)
    if st.button("Send"):
        response = conversation.invoke(user_question)
        message = {"human": user_question, "AI": response["response"]}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response["response"])

        # Add a scrollable container for the chat history
        chat_history_container = st.container()
        chat_history_container.write("Chat History:")
        for message in st.session_state.chat_history:
            chat_history_container.write(f"{message['human']}: {message['AI']}")

        # Using SQLAlchemy to interact with the database
        with Session() as session:
            # Serialize chat history as a JSON string to store in the database
            new_chat_history = ChatHistory(chat_history=json.dumps(st.session_state.chat_history))
            session.add(new_chat_history)
            session.commit()
