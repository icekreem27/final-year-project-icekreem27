#!/usr/bin/env python -B

import os
from openai import OpenAI
import sys
from pathlib import Path

# ------- To import functions needed from Tools -------
# Get the path of the current file
project_root = Path(__file__).resolve().parent.parent
# Add the project root to the sys.path if not already added
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Tools.embedding import create_embeddings, save, load_embeddings, ask

# Initialize the OpenAI client with the API key
# (stored as env variable)
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# Set paths
file_path = "Datasets/embeddings.csv"
folder_paths = ['Data Mining files/pdf_txt_files', 'Data Mining files/pptx_txt_files']

if __name__ == "__main__":
    # Check if embeddings exist, otherwise create them
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Creating embeddings...")
        
        # Create embeddings from module data in the folders
        df = create_embeddings(folder_paths)
        
        # Save the created DataFrame to CSV
        save(df, file_path)
        print(f"Embeddings saved to {file_path}.")
    else:
        # Else load if existing
        print(f"Loading embeddings from {file_path}...")
        df = load_embeddings(file_path)  
            
    # Conversation loop
    while True:
        user_input = input("User> ")
        response = ask(user_input, df)
        print("Model> ", response)