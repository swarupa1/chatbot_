!pip install -q -U sentence-transformers bitsandbytes accelerate

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# ---------------------------
# Hugging Face Token
# ---------------------------
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------
# Knowledge Base & Vector Store
# ---------------------------
documents = [
    """ShopSmart Return Policy\nProducts can be returned within 30 days.\nDamaged products are eligible for replacement or refund.\nFinal sale products cannot be returned.\nRefunds above $500 require human review.\n""",
    """Premium Membership\nFree shipping\nPriority support\nEarly access sales\n""",
    """Payment Methods\nVisa\nMasterCard\nPayPal\nApple Pay\n"""
]

# Load embedding model
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
doc_embeddings = embed_model.encode(documents)

def simple_search(query, k=1):
    query_embedding = embed_model.encode([query])
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten() / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding))
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [documents[i] for i in top_indices]

# ---------------------------
# Hugging Face Model (Zephyr 7B)
# ---------------------------
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
print("Loading LLM in 4-bit (this may take a few minutes)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# Modern 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        token=HF_TOKEN, 
        device_map="auto", 
        quantization_config=bnb_config
    )
except Exception as e:
    print(f"Falling back to float16: {e}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN, torch_dtype=torch.float16, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ---------------------------
# RAG Function
# ---------------------------
def get_answer(query):
    try:
        relevant_docs = simple_search(query, k=1)
        context = "\n".join(relevant_docs)
        prompt = f"<|system|>\nYou are a customer support chatbot. Use ONLY the provided context to answer.\nContext:\n{context}</s>\n<|user|>\n{query}</s>\n<|assistant|>"
        # Zephyr-specific generation settings
        response = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_p=0.9)
        return response[0]["generated_text"].split("<|assistant|>")[-1].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------
# Test Execution
# ---------------------------
if __name__ == "__main__":
    test_query = "What is your return policy?"
    print(f"\nCustomer: {test_query}")
    print(f"Agent: {get_answer(test_query)}")