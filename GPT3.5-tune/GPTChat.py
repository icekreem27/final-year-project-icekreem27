from openai import OpenAI
import os

# Initialize the OpenAI client with the API key
# (stored as env variable)
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

while True:
    user_input = input("User> ")
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::948INxw3", # model ID
        messages=[
            {"role": "system", "content": "You are a factual chatbot that will help student learn about the COMP2121 Data Mining module."},
            {"role": "user", "content": user_input}
        ]
    )
    print("model> ", response.choices[0].message.content)