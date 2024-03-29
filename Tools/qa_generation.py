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
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def spacy_text_splitter(text, chunk_size=80):
    nlp = spacy.load("en_core_web_sm")
    nlp.disable_pipes('ner', 'parser')
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0
    for sent in doc.sents:
        sent_token_count = len(sent)
        if current_chunk_token_count + sent_token_count > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_token_count = 0
        current_chunk.append(sent.text)
        current_chunk_token_count += sent_token_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
chat = ChatOpenAI(temperature=0.6)

def process_folder(folder_path, output_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    all_qa = []
    for file_name in tqdm(files, desc="Processing files"):
        try:
            file_path = os.path.join(folder_path, file_name)
            loader = TextLoader(file_path)
            doc = loader.load()[0]
            cleaned_content = clean_text(doc.page_content)
            texts = spacy_text_splitter(cleaned_content)
            chain = QAGenerationChain.from_llm(chat)
            for text_chunk in texts:
                qa = chain.run(text_chunk)
                all_qa.extend(qa)
        except json.JSONDecodeError as e:
            pass
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_qa, json_file, ensure_ascii=False, indent=4)

# Specify the folder paths and output files
folders_and_outputs = [
    ("Data Mining files/pptx_txt_files", "Datasets/QA_Pairs/pptx_QA_Pairs.json"),
    ("Data Mining files/pdf_txt_files", "Datasets/QA_Pairs/pdf_QA_Pairs.json")
]

# Process each folder
for folder_path, output_path in folders_and_outputs:
    process_folder(folder_path, output_path)
