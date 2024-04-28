import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "tell a joke",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)
