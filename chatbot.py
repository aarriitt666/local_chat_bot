import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from settings import MAX_BOT_MEMORY_SIZE, LLM_MOD, MAX_CHAT_HISTORY_SIZE
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from database import Session
from models import ChatHistory
import os
import json
import datetime  # For adding timestamps

st.title("Chat with Groq!")
st.write("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

def load_chat_history():
    with Session() as session:
        last_entries = session.query(ChatHistory).order_by(ChatHistory.id.desc()).limit(MAX_CHAT_HISTORY_SIZE).all()
        chat_history = [json.loads(entry.chat_history) for entry in last_entries]
        return chat_history[::-1]  # Reverse to display oldest first

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

memory = ConversationBufferWindowMemory(k=MAX_BOT_MEMORY_SIZE)
for message in st.session_state.chat_history:
    if isinstance(message, dict) and "human" in message and "AI" in message:
        memory.save_context({"input": message["human"]}, {"output": message["AI"]})
    else:
        print("Message format error:", message)

groq_chat = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=LLM_MOD, temperature=0.5)
conversation = ConversationChain(llm=groq_chat, memory=memory)

def send_message():
    formatted_input = st.session_state.user_input.replace("\n", "\n\n")  # Double newlines for proper Markdown paragraphs
    response = conversation.invoke(formatted_input)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = {"human": f"You ({timestamp}): {formatted_input}", "AI": f"Bot ({timestamp}): {response['response']}"}
    st.session_state.chat_history.append(message)
    st.session_state.user_input = ""  # Clear the input after processing

    with Session() as session:
        new_entry = ChatHistory(chat_history=json.dumps(message))
        session.add(new_entry)
        session.commit()

with st.expander("Chat", expanded=True):
    user_input = st.text_area("Ask a question:", height=200, value=st.session_state.user_input if 'user_input' in st.session_state else "", key="user_input")
    send_button = st.button("Send", on_click=send_message)

    chat_history_container = st.container()
    chat_history_container.write("Chat History:")
    for message in reversed(st.session_state.chat_history):
        if isinstance(message, dict) and "human" in message and "AI" in message:
            chat_history_container.markdown(f"**{message['human']}**\n\n_{message['AI']}_")  # Ensure newline handling
