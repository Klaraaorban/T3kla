import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "polar_code_v1_0"
EMBEDDING_MODEL_PATH = "all-mpnet-base-v2"
DATA_DIR = "data"

PDF_FILES = [
    os.path.join(DATA_DIR, "MSC.385(94)_Polar_Code_Safety.pdf"),
    os.path.join(DATA_DIR, "MEPC.264(68)_Polar_Code_Env.pdf")
]

all_documents = []
for file_path in PDF_FILES:
    if os.path.exists(file_path):
        print(f"Loading document: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = os.path.basename(file_path)
        all_documents.extend(docs)
    else:
        print(f"WARNING: File not found at {file_path}. Skipping.")
        
if not all_documents:
    print("FATAL: No documents loaded. Please check the DATA_DIR path.")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    separators=["\n\n", "\n", ".", "!", "?"]
)
texts = text_splitter.split_documents(all_documents)
print(f"Split {len(all_documents)} document pages into {len(texts)} searchable chunks.")

print(f"Initializing embedding model: {EMBEDDING_MODEL_PATH}...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

print(f"Creating/Updating ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}...")
db = Chroma.from_documents(
    texts, 
    embeddings, 
    persist_directory=CHROMA_DIR, 
    collection_name=COLLECTION_NAME
)
db.persist()
print("Ingestion complete. ChromaDB persistence successful.")