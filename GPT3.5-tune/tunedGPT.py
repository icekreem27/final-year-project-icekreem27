from openai import OpenAI

client = OpenAI()

MODEL_ID = "ft:gpt-3.5-turbo-0613:personal::8pyiv0AN"

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain the difference between reverse and interpolation in language modeling?"}
    ]
)

# Print the response message
print(response.choices[0].message)
