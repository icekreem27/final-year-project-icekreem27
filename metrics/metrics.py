import os
import sys
import json
import numpy as np

from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import Counter
from datasets import load_metric

from scipy.spatial import distance
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu

# Get the path of the current file
project_root = Path(__file__).resolve().parent.parent
# Add the project root to the sys.path if not already added
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
from Tools.embedding import create_embeddings, save, load_embeddings, ask, generate_embeddings, ask_no_embeddings

# Load metrics that are appropriate for evaluating QA models
rouge = load("rouge")
bleu = load("bleu")
meteor = load("meteor")
bertscore = load("bertscore", model_type="bert-base-multilingual-cased")

# Paths for required resources
embedding_path = "Datasets/embeddings.csv"
folder_paths = 'Data Mining files/pdf_txt_files', 'Data Mining files/pptx_txt_files'
QA_path = "Datasets/Augmented/validation_dataset.jsonl"

# Initialize variable for cumulative scores
cumulative_scores = {"rouge": 0, "bleu": 0, "meteor": 0, "bertscore": 0, "f1_score_result": 0, "semantic_similarity_score": 0}
query_count = 0

def f1_score(prediction: str, truth: str) -> float:
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def semantic_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    # Convert embeddings to NumPy arrays and ensure they are 1-D
    vector1 = np.array(embedding1).flatten()
    vector2 = np.array(embedding2).flatten()
    
    # Calculate and return the cosine similarity
    return 1 - distance.cosine(vector1, vector2)

def evaluate_model(model):
    global query_count
    # Check if embeddings exist, otherwise create them
    if not os.path.exists(embedding_path):
        print(f"{embedding_path} not found. Creating embeddings...")
        df = create_embeddings(folder_paths)
        save(df, embedding_path)
        print(f"Embeddings saved to {embedding_path}.")
    else:
        print(f"Loading embeddings from {embedding_path}...")
        df = load_embeddings(embedding_path)
    
    with open(QA_path, 'r', encoding='utf-8') as file:
        lines = []
        for line in file:
            try:
                # Attempt to parse the line as JSON
                data = json.loads(line)
                # Append the relevant content to the questions list
                lines.append(data)
            except json.JSONDecodeError:
                # If an error occurs, skip this line
                continue
    
    for line in tqdm(lines, desc="Evaluating", unit="line"):
            prompt = data['messages'][1]['content']
            answer = data['messages'][2]['content']
            
            generated_text = ask(prompt, df, model)
            # generated_text = ask_no_embeddings(prompt)

            
            # Evaluate with each metric and update cumulative scores
            update_scores(generated_text, answer)
            query_count += 1

    print_average_scores()

def update_scores(generated_text, reference):
    # No need to tokenize here for BLEU, as the evaluate library expects strings.
    generated_text_str = ' '.join(generated_text)  # If generated_text is a list of tokens.
    reference_str = ' '.join(reference)  # If reference is a list of tokens.

    # Now, the input is a single string for both prediction and reference.
    rouge_score = rouge.compute(predictions=[generated_text], references=[reference])
    bleu_score = bleu.compute(predictions=[generated_text_str], references=[reference_str])
    meteor_score = meteor.compute(predictions=[generated_text_str], references=[reference_str])
    bertscore_result = bertscore.compute(predictions=[generated_text_str], references=[reference_str], lang="en")


    prediction_embedding = generate_embeddings(generated_text)
    reference_embedding = generate_embeddings(reference)
    
    # Accumulate scores
    cumulative_scores["rouge"] += rouge_score['rougeL']
    cumulative_scores["bleu"] += bleu_score['bleu']
    cumulative_scores["meteor"] += meteor_score["meteor"]
    cumulative_scores["bertscore"] += bertscore_result["f1"][0]
    cumulative_scores["f1_score_result"] += f1_score(generated_text, reference)
    cumulative_scores["semantic_similarity_score"] += semantic_similarity(prediction_embedding, reference_embedding)
    
    # print(f"Metrics for Query {query_count + 1}:")
    # print(f"Rouge: {rouge_score['rougeL']:.4f}")
    # print(f"Bleu: {bleu_score['bleu']:.4f}")
    # print(f"Meteor: {meteor_score['meteor']:.4f}")
    # print(f"Bertscore: {bertscore_result['f1'][0]:.4f}")
    # print(f"F1 Score: {f1_score(generated_text, reference):.4f}")
    # print(f"Semantic Similarity: {semantic_similarity(prediction_embedding, reference_embedding):.4f}")

    

def print_average_scores():
    print("Average Metrics:")
    for metric, score in cumulative_scores.items():
        average_score = score / query_count
        print(f"{metric.capitalize()}: {average_score:.4f}")

if __name__ == "__main__":
    model = "ft:gpt-3.5-turbo-0125:personal::90bpxw0O"
    evaluate_model(model)
