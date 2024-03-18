import os
import json
import spacy
import warnings
import re
from tqdm import tqdm  # for progress bar
from langchain_community.document_loaders import TextLoader
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_openai import ChatOpenAI
from langchain.chains import QAGenerationChain

def clean_text(text):
    # Regex, keeps english letters, numbers, punctuation, and whitespace
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def spacy_text_splitter(text, chunk_size=400):
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Disable unnecessary components to speed up processing
    nlp.disable_pipes('ner', 'parser')
    
    # Add the 'sentencizer' component to the pipeline
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")
        
    # Process the text
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0
    
    for sent in doc.sents:
        sent_token_count = len(sent)
        # Check if adding this sentence would exceed the chunk size
        if current_chunk_token_count + sent_token_count > chunk_size:
            # If the current chunk is not empty, add it to the list
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_token_count = 0
        # Add the sentence to the current chunk
        current_chunk.append(sent.text)
        current_chunk_token_count += sent_token_count
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Explicitly disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Filter and disable deprecation warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


# Set the chat model using LangChain's OpenAI integration
chat = ChatOpenAI(temperature=0)

# Specify the folder path containing your text files
folder_path = "Data Mining files/pptx_txt_files"
files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
all_qa = []

# Loop through all files in the directory
for file_name in tqdm(files, desc="Processing files"):
    try:
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Load the document
        loader = TextLoader(file_path)
        doc = loader.load()[0]  # Assuming each file contains a single document

        # Clean before further processing
        cleaned_content = clean_text(doc.page_content)

        # Split the cleaned text into fixed chunks
        texts = spacy_text_splitter(cleaned_content)

        # Initialize the Q&A Generation Chain directly with LangChain's OpenAI integration
        chain = QAGenerationChain.from_llm(chat)        
        
        # Process each chunk of text
        doc_qa = []
        for text_chunk in texts:
            # Generate Q&A for the chunk
            qa = chain.run(text_chunk)
            all_qa.extend(qa)

    # Error handling
    except json.JSONDecodeError as e:
        pass
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Save the Q&A pairs to the JSON file
output_path = "Datasets/QA_Pairs/pptx_QA_Pairs.json"
with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(all_qa, json_file, ensure_ascii=False, indent=4)
