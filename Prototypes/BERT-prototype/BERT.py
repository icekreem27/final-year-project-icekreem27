from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import json

# Load the fine-tuned model and tokenizer
model = BertForQuestionAnswering.from_pretrained('./results/my_fine_tuned_model')
tokenizer = BertTokenizer.from_pretrained('./results/my_fine_tuned_model')


# Function to read data from dataset.json
def read_data(file_path='dataset.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Simple context retrieval based on first matching paragraph
# This function needs improvement for real-world applications
def find_context(question, data):
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            # Check if any question in the paragraph matches the user question or use other heuristics
            for qa in paragraph['qas']:
                if question.lower() in qa['question'].lower():  # Simple heuristic
                    return context
    return "No relevant context found."  # Fallback

# Function to ask question
def ask_question(context, question):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

# Load dataset
data = read_data('dataset.json')  # Adjust this path

print("Enter 'quit' to exit.")
while True:
    question = input("Ask a question: ")
    if question.lower() == 'quit':
        break
    
    context = find_context(question, data)
    if context == "No relevant context found.":
        print(context)
        continue
    
    answer = ask_question(context, question)
    print(f"Answer: {answer}\n")
