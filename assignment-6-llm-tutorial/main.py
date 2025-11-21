import getpass

# Get the OpenAI API key and confirm it is set
openai_key = getpass.getpass("MY_API_KEY: ")
if openai_key and openai_key.strip():
    print("OPENAI_KEY is non-empty")
else:
    print("OPENAI_KEY is empty or not set")


