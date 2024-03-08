import os
import json
import logging
from tqdm import tqdm  # for progress bar
from milvus import default_server
from pymilvus import connections, utility, db
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import SpacyTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import QAGenerationChain
from gptcache import cache
from gptcache.adapter.langchain_models import LangChainChat
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


# Remove exact duplicate QA pairs
def removeDuplicates(all_qa):
    unique_qa = []
    seen = set()
    for item in all_qa:
        qa_pair = (item['question'], item['answer'])
        if qa_pair not in seen:
            unique_qa.append(item)
            seen.add(qa_pair)

# Define a function to extract the content from the prompt for caching
def get_msg_func(data, **_):
    return data.get("messages")[-1].content

# Text cleaning to preprocess the document content
def clean_text(text):
    # Replace or remove control characters
    cleaned_text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

# Explicitly disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings from the milvus library
# logging.getLogger('milvus').setLevel(logging.ERROR)  # Only show errors, no warnings

# Start the default Milvus server
default_server.start()

# Delete existing cache and datasbase if needed
# conn = connections.connect(host="127.0.0.1", port=19530)
# utility.drop_collection("gptcache")
# db.drop_database("sqlite")
print(db.list_database())


# Initialize new GPTCache with specified configurations
onnx = Onnx()
cache_base = CacheBase('sqlite')
vector_base = VectorBase('milvus', host='127.0.0.1', port='19530', dimension=onnx.dimension)
data_manager = get_data_manager(cache_base, vector_base)

cache.init(
    pre_embedding_func=get_msg_func,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)
cache.set_openai_key()

# Text splitter with specific settings - chunk size
text_splitter = SpacyTextSplitter(chunk_size=512)

# Set chat model
chat = LangChainChat(chat=ChatOpenAI(temperature=1.4))

# Process folders
folder_path = "Data Mining files/pdf_txt_files"
files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
all_qa = []
print(db.list_database())
print(utility.list_collections)

# Loop through all files in the directory
for file_name in tqdm(files, desc="Processing files"):
    try:
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Load the document
        loader = TextLoader(file_path)
        doc = loader.load()[0]  # Assuming each file contains a single document

        # Clean text
        cleaned_text = clean_text(doc.page_content)

        # Split the cleaned text into fixed chunks
        texts = text_splitter.split_text(cleaned_text)

        # Initialize the Q&A Generation Chain
        chain = QAGenerationChain.from_llm(chat)
        
        doc_qa = []
        # Process each chunk of text
        for text_chunk in texts:
            # Generate Q&A for the current chunk and then append it to the doc_qa list
            qa = chain.run(text_chunk)
            doc_qa.extend(qa)

        # Then append the collected Q&A pairs to the overall list
        all_qa.extend(doc_qa)
        
    except json.JSONDecodeError as e:
        print(f"Error processing file {file_name}: JSON decoding error - {e}")
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        
        
# Set directory and output
output_path = "Datasets/QA_Pairs/pdf_QA_Pairs.json"

# Remove duplicated pairs
final_qa = removeDuplicates(all_qa)

# Save the Q&A pairs to the JSON file
with open(output_path, 'x', encoding='utf-8') as json_file:
    json.dump(final_qa, json_file, ensure_ascii=False, indent=4)