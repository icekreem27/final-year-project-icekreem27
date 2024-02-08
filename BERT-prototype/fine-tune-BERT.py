from transformers import BertTokenizer, BertForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
import json

# Function to read data from a JSON file
def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Convert JSON data to Hugging Face Dataset
def json_to_dataset(data):
    questions, contexts, answers = [], [], []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                questions.append(qa['question'])
                contexts.append(paragraph['context'])
                answers.append({'text': [answer['text'] for answer in qa['answers']], 
                                'answer_start': [answer['answer_start'] for answer in qa['answers']]})
    return Dataset.from_dict({
        'question': questions,
        'context': contexts,
        'answers': answers
    })


# Function to preprocess the dataset for question answering
def preprocess_function(examples):
    questions = [q.strip() for q in examples['question']]
    contexts = [c.strip() for c in examples['context']]
    answers = examples['answers']
    inputs = tokenizer(questions, contexts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    start_positions = [context.find(answer['text'][0]) if answer['text'][0] in context else 0 for answer, context in zip(answers, contexts)]
    end_positions = [start + len(answer['text'][0]) for start, answer in zip(start_positions, answers)]
    inputs.update({'start_positions': start_positions, 'end_positions': end_positions})
    return inputs


# Load the dataset
file_path = 'augmented_squad_dataset_synonym_back_translation.json'
raw_data = read_data(file_path)
processed_data = json_to_dataset(raw_data)

# Initialize the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Preprocess the dataset
tokenized_datasets = processed_data.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Directory where the model predictions and checkpoints will be written
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=8,   # Batch size per GPU/TPU core/CPU for training
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./results/my_fine_tuned_model')
tokenizer.save_pretrained('./results/my_fine_tuned_model')
