import os
import json
from tqdm import tqdm  # for progress bar
from milvus import default_server
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import QAGenerationChain
from gptcache import cache
from gptcache.adapter.langchain_models import LangChainChat
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Explicitly disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Start the default Milvus server
default_server.start()

# Define a function to extract the content from the prompt for caching
def get_msg_func(data, **_):
    return data.get("messages")[-1].content

# Initialize the GPTCache with specified configurations
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

# Text splitter with specific settings - 1500 chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=400, chunk_size=1500)

# Set chat model
chat = LangChainChat(chat=ChatOpenAI(temperature=0))

# Process folders
folder_path = "Data Mining files/pdf_txt_files"
files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
all_qa = []

for file_name in tqdm(files, desc="Processing files"):
    file_path = os.path.join(folder_path, file_name)
    loader = TextLoader(file_path)
    doc = loader.load()[0]  # Assuming each file contains a single document
    chain = QAGenerationChain.from_llm(chat, text_splitter=text_splitter)
    qa = chain.run(doc.page_content)
    all_qa.extend(qa)

# Set directory and output
output_dir = "/Datasets/QA_Pairs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "pdf_QA_Pairs.json")

# Save the Q&A pairs to the JSON file
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(all_qa, json_file, ensure_ascii=False, indent=4)