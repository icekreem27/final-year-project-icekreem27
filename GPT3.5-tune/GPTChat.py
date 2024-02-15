from openai import OpenAI
import os

# Initialize the OpenAI client with the API key
# (stored as env variable)
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# system message to initialize the context for the AI
system_message = {"role": "system", "content": "You are a factual chatbot that will help student learn about the COMP2121 Data Mining module."}

while True:
    user_input = input("User> ")
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:personal::8sUn174X", # model ID
        messages=[
            system_message,
            {"role": "user", "content": user_input}
        ]
    )
    print("model> ", response.choices[0].message.content)