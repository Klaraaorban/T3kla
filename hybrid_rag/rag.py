from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse 
from pathlib import Path
from starlette.concurrency import run_in_threadpool
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import httpx 
import time
import numpy as np
from rank_bm25 import BM25Okapi 
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

app = FastAPI()

HTML_FILE_PATH = Path("static.html")

class Query(BaseModel):
    question: str

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "polar_code_v1_0"
EMBEDDING_MODEL = "all-mpnet-base-v2" 

MODEL_ID = "microsoft/phi-3.5-mini-instruct" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TOP_K_INITIAL_RETRIEVAL = 15
TOP_K_FINAL = 3
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

tokenizer = None
model_pipeline = None
reranker = None
chroma_client = None
collection = None
embeddings = None
query_cache: Dict[str, Any] = {}
bm25_index: Union[BM25Okapi, None] = None
bm25_document_map: Dict[str, Any] = {} 


conversation_history: List[Dict[str, str]] = []

@app.on_event("startup")
async def startup_event():
    global tokenizer, model_pipeline, reranker, chroma_client, collection, embeddings
    global bm25_index, bm25_document_map
    
    print(f"Loading LLM: {MODEL_ID} to {DEVICE}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            return_full_text=False 
        )
        print("Phi-3.5 Mini Instruct loaded successfully.")
    except Exception as e:
        print(f"Error loading Phi-3.5 model: {e}")
        
    print("Loading RAG components...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=None) 
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device=str(DEVICE))
        
        print(f"Chroma collection '{COLLECTION_NAME}' loaded.")
        print(f"Reranker model '{RERANKER_MODEL}' loaded.")
    except Exception as e:
        print(f"Error loading RAG components: {e}")

    print("Building BM25 Index for Hybrid Search...")
    try:
        all_data = collection.get(include=['documents', 'metadatas'])
        
        corpus = all_data['documents']
        
        bm25_tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25_index = BM25Okapi(bm25_tokenized_corpus)
        
        bm25_document_map = {
            doc_id: {"content": doc, "metadata": meta}
            for doc_id, doc, meta in zip(all_data['ids'], all_data['documents'], all_data['metadatas'])
        }
        print(f"BM25 Index built with {len(corpus)} documents.")
    except Exception as e:
        print(f"Error building BM25 Index: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_static_html():
    if not HTML_FILE_PATH.exists():
        return HTMLResponse(content="<h1>Error: static.html not found.</h1>", status_code=404)
    with open(HTML_FILE_PATH, 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

def ask_phi3_local(query: str, chunks: List[Dict[str, Any]], mode: str = "rag", conversation_history: List[Dict[str, str]] = None) -> str:
    if not model_pipeline:
        return "Error: Language model not loaded."
    
    if mode == "chitchat":
        system_prompt = "You are a helpful and kind AI assistant. Respond to the user's greeting or casual question concisely and naturally. Your name is T3kla!"
        user_query = query
        context = "" 
    else:
        context_entries = []
        for i, c in enumerate(chunks):
            score = f" (Rerank Score: {c.get('score', 0.0):.4f})" if c.get('score') is not None else ""
            content = c['content']
            source = c.get('metadata', {}).get('source', 'unknown')
            page = c.get('metadata', {}).get('page', 'n/a')
            context_entries.append(f"Chunk {i+1}{score}:\n{content}\n(Source: {source}, Page: {page})")
            
        context = "\n\n---\n\n".join(context_entries)
        
        system_prompt = "You are a strict, factual maritime safety analyst. Your sole purpose is to answer the user's question using ONLY the provided text in the Context. Your answer MUST be directly supported by the information presented below. If the answer cannot be fully supported by the Context, you MUST output the following exact phrase and nothing else: 'Data is insufficient to answer the question.'"
        
        user_query = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    if conversation_history is not None:
        for msg in conversation_history:
            messages.append(msg)

    messages.append({"role": "user", "content": user_query})

    prompt = model_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        outputs = model_pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=model_pipeline.tokenizer.eos_token_id
        )
        
        response_text = outputs[0]['generated_text']
        if response_text.startswith(prompt):
             return response_text[len(prompt):].strip()
        else:
             return response_text.strip()
            
    except Exception as e:
        return f"An error occurred during local model generation: {e}"

@app.post("/ask")
async def ask_endpoint(query: Query):
    global conversation_history
    start_time = time.time()
    answer = "Error: Internal server processing failed before completion."
    final_chunks = []
    mode = "rag"

    if len(conversation_history) > 10:
        conversation_history = []

    if not embeddings or not collection or not reranker or not bm25_index:
        return {"answer": "System components are still loading or failed to initialize.", "runtime_sec": 0.0, "cached": False}

    if query.question in query_cache and not conversation_history:
        answer = query_cache[query.question]
        runtime = time.time() - start_time
        return {"answer": answer, "runtime_sec": runtime, "cached": True}
    
    current_user_message = {"role": "user", "content": query.question}
    is_chitchat = len(query.question.split()) <= 4
    
    if is_chitchat:
        mode = "chitchat"
        # FIX 1: Pass history to the chitchat mode
        answer = await run_in_threadpool(ask_phi3_local, query.question, final_chunks, mode, conversation_history=conversation_history)
        
    else:
        mode = "rag"
        augmented_query = query.question
        
        if conversation_history:
            history_context_str = " ".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
            augmented_query = f"{history_context_str} USER: {query.question}"
            if len(augmented_query) > 512:
                augmented_query = augmented_query[-512:]
        
        try:
            query_embedding = await run_in_threadpool(embeddings.embed_query, augmented_query)
            
            results_dense = await run_in_threadpool(
                collection.query,
                query_embeddings=[query_embedding], 
                n_results=TOP_K_INITIAL_RETRIEVAL,
                include=['documents', 'metadatas']
            )
            
            all_candidates = {}
            for doc_id, c, m in zip(results_dense["ids"][0], results_dense["documents"][0], results_dense["metadatas"][0]):
                all_candidates[doc_id] = {"content": c, "metadata": m}
            
            tokenized_query = augmented_query.split(" ")
            bm25_scores = bm25_index.get_scores(tokenized_query)
            
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K_INITIAL_RETRIEVAL]

            all_data_ids = list(bm25_document_map.keys())
            for idx in top_bm25_indices:
                doc_id = all_data_ids[idx]
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = bm25_document_map[doc_id]
            
            candidate_list = list(all_candidates.values())
            
            # FIX 2: Use the augmented_query for RERANKING
            pairs = [[augmented_query, chunk['content']] for chunk in candidate_list]
            
            scores = await run_in_threadpool(reranker.predict, pairs)
            
            for i, chunk in enumerate(candidate_list):
                chunk['score'] = scores[i]
        
            candidate_list.sort(key=lambda x: x['score'], reverse=True)
            final_chunks = candidate_list[:TOP_K_FINAL]

            # FIX 3: Use the augmented_query as the question for LLM prompt
            answer = await run_in_threadpool(ask_phi3_local, augmented_query, final_chunks, mode, conversation_history=conversation_history)
        
        except Exception as e:
            answer = f"A critical error occurred during retrieval or processing: {e}"
            print(f"Critical Error: {e}")
            
    end_time = time.time()
    runtime = end_time - start_time

    conversation_history.append(current_user_message)
    conversation_history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "runtime_sec": runtime,
        "cached": False,
        "context_chunks": [
            {"content": c['content'], "source": c['metadata'].get('source', 'unknown'), "page": c['metadata'].get('page', 'n/a'), "score": f"{c['score']:.4f}"} for c in final_chunks
        ]
    }