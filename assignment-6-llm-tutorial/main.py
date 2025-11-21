import os

openai_key = os.getenv('OPENAI_KEY')
if openai_key and openai_key.strip():
    print("OPENAI_KEY is non-empty")
else:
    print("OPENAI_KEY is empty or not set")
