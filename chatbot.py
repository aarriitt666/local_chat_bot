import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from settings import MAX_BOT_MEMORY_SIZE, LLM_MOD
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
        last_entry = session.query(ChatHistory).order_by(ChatHistory.id.desc()).first()
        if last_entry:
            try:
                loaded_data = json.loads(last_entry.chat_history)
                if all("human" in message and "AI" in message for message in loaded_data):
                    return loaded_data
                else:
                    print("Loaded data does not match expected format.")
                    return []
            except json.JSONDecodeError as e:
                print("Failed to decode JSON:", str(e), "Data:", last_entry.chat_history)
                return []
        return []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

memory = ConversationBufferWindowMemory(k=MAX_BOT_MEMORY_SIZE)
for message in st.session_state.chat_history:
    if "human" in message and "AI" in message:
        memory.save_context({"input": message["human"]}, {"output": message["AI"]})
    else:
        print("Message format error:", message)

groq_chat = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=LLM_MOD, temperature=0.5)
conversation = ConversationChain(llm=groq_chat, memory=memory)

def send_message():
    response = conversation.invoke(st.session_state.user_input)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = {"human": f"You ({timestamp}): {st.session_state.user_input}", "AI": f"Bot ({timestamp}): {response['response']}"}
    st.session_state.chat_history.append(message)
    st.session_state.user_input = ""  # Clear the input after processing

with st.expander("Chat", expanded=True):
    user_input = st.text_area("Ask a question:", height=200, value=st.session_state.user_input if 'user_input' in st.session_state else "", key="user_input")
    send_button = st.button("Send", on_click=send_message)

    chat_history_container = st.container()
    chat_history_container.write("Chat History:")
    # Display only the last 19 messages
    for message in reversed(st.session_state.chat_history[-19:]):
        chat_history_container.markdown(f"**{message['human']}**")
        chat_history_container.markdown(f"_{message['AI']}_")  # Italicize bot responses for clarity

    with Session() as session:
        new_chat_history = ChatHistory(chat_history=json.dumps(st.session_state.chat_history))
        session.add(new_chat_history)
        session.commit()
