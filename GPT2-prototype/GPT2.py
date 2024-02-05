from transformers import GPT2Tokenizer, GPT2LMHeadModel

def setup_chatbot():
    # Load pre-trained model tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    return tokenizer, model

def chatbot_response(tokenizer, model, message, max_length=50):
    # Encode input text
    input_ids = tokenizer.encode(message, return_tensors='pt')

    # Generate response
    chat_output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # Decode and print the response
    chat_response = tokenizer.decode(chat_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return chat_response

# Setup the chatbot
tokenizer, model = setup_chatbot()

# Chat
print("Hello, type something to begin...")

while True:
    try:
        user_input = input("User> ")
        if user_input.lower() == 'bye':
            print("GPT2> Bye!")
            break
        response = chatbot_response(tokenizer, model, user_input)
        print(f"GPT2> {response}")
    except (KeyboardInterrupt, EOFError, SystemExit):
        print("\GPT2> Bye!")
        break
