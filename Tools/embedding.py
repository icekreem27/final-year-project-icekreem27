from typing import List, Tuple
import os
import ast
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy import spatial
from openai import OpenAI

# Constants for easy configuration and adjustment
MAX_CHUNK_TOKENS = 125
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 1000

# Initialize the OpenAI client for API requests
client = OpenAI()

def split_into_chunks(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """Splits a text into chunks of specified maximum token length."""
    tokens = text.split()
    for i in range(0, len(tokens), max_tokens):
        yield ' '.join(tokens[i:i+max_tokens])

def read_text_file(file_path: str) -> str:
    """Reads and returns the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of text chunks using the OpenAI API."""
    embeddings = []
    for i in range(0, len(text_chunks), BATCH_SIZE):
        batch = text_chunks[i:i+BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

def create_embeddings(folder_paths: List[str]) -> pd.DataFrame:
    """Creates a DataFrame of text chunks and their embeddings for all text files in the specified folders."""
    text_chunks = []
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                text_chunks.extend(list(split_into_chunks(read_text_file(file_path))))
    embeddings = generate_embeddings(text_chunks)
    return pd.DataFrame({"text": text_chunks, "embedding": embeddings})

def save(df: pd.DataFrame, file_path: str):
    """Saves the DataFrame with embeddings to a CSV file."""
    df['embedding'] = df['embedding'].apply(str)
    df.to_csv(file_path, index=False)

def load_embeddings(csv_file_path: str) -> pd.DataFrame:
    """Loads embeddings from a CSV file into a DataFrame."""
    df = pd.read_csv(csv_file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    return df

def texts_ranked_by_relatedness(query: str, df: pd.DataFrame, top_n: int = 5) -> Tuple[List[str], List[float]]:
    query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_embedding = np.array(query_embedding_response.data[0].embedding)

    # Calculate relatedness scores using a list
    texts_and_relatedness = []
    for _, row in df.iterrows():
        row_embedding = np.array(row['embedding'])
        texts_and_relatedness.append((row['text'], 1 - distance.cosine(query_embedding, row_embedding)))

    # Sort texts by relatedness score in descending order
    texts_and_relatedness.sort(key=lambda x: x[1], reverse=True)

    # Unpack texts and relatedness scores into separate lists, limited to top_n items
    texts, relatedness_scores = zip(*texts_and_relatedness[:top_n])

    # Return texts and relatedness
    return list(texts), list(relatedness_scores)

def query_message(query: str, df: pd.DataFrame, model: str, token_budget: int) -> str:
    """Constructs a message for GPT, incorporating relevant texts from the DataFrame."""
    strings, _ = texts_ranked_by_relatedness(query, df)
    message = 'Use the below lecture contents to answer the subsequent question. If the answer cannot be found in the contents, write "I could not find an answer."'
    for string in strings:
        if len(message) + len(string) + len(query) + 3 <= token_budget:
            message += f'\n\nLecture content section:\n"""\n{string}\n"""'
        else:
            break
    return message + f"\n\nQuestion: {query}"

def ask(query: str, df: pd.DataFrame, model: str = "gpt-3.5-turbo", token_budget: int = 4096 - 500, print_message: bool = False) -> str:
    """Generates a response to a user query using a specified GPT model and a DataFrame of embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are a knowledgeable assistant that will help students learn about the COMP2121 Data Mining module."}, {"role": "user", "content": message}], temperature=0)
    return response.choices[0].message.content

if __name__ == "__main__":
    file_path = "Datasets/embeddings.csv"
    
    # Check if embeddings exist, otherwise create them
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Creating embeddings...")
        # Define the folder path where your text files are located
        folder_paths = ['Data Mining files/pdf_txt_files', 'Data Mining files/pptx_txt_files']
        # Create embeddings from text files in the specified folder
        df = create_embeddings(folder_paths)
        # Save the created DataFrame to CSV
        save(df, file_path)
        print(f"Embeddings saved to {file_path}.")
    else:
        print(f"Loading embeddings from {file_path}...")
        df = load_embeddings(file_path)  
          
    while True:
        user_input = input("User> ")
        response = ask(user_input, df)
        print("Model> ", response)
