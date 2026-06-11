import os
import sys
import subprocess

# Auto-install missing dependencies
required_packages = {
    'chromadb': 'chromadb',
    'transformers': 'transformers',
    'sentence_transformers': 'sentence-transformers',
    'torch': 'torch',
    'bitsandbytes': 'bitsandbytes'
}

for module, package in required_packages.items():
    try:
        __import__(module)
    except ImportError:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package], 
                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}, attempting without --user flag...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                    stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e2:
                print(f"Error: Could not install {package}: {e2}")
                sys.exit(1)

import chromadb
import torch
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# Hugging Face Token
# ---------------------------

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------
# Knowledge Base
# ---------------------------

documents = [
    """ShopSmart Return Policy
Products can be returned within 30 days.
Damaged products are eligible for replacement or refund.
Final sale products cannot be returned.
Refunds above $500 require human review.
""",
    """Premium Membership
Free shipping
Priority support
Early access sales
""",
    """Payment Methods
Visa
MasterCard
PayPal
Apple Pay
"""
]

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

db = chromadb.Client()

collection = db.get_or_create_collection(
    name="customer_support",
    embedding_function=embedding_function
)

collection.add(
    ids=["1", "2", "3"],
    documents=documents
)

# ---------------------------
# Hugging Face Model
# ---------------------------

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200
)

# ---------------------------
# RAG Function
# ---------------------------

def get_answer(query):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=2
        )

        context = "\n".join(results["documents"][0]) if results["documents"] else "No relevant context found."

        prompt = f"""You are a customer support chatbot.

Use ONLY the context below to answer the customer's question.

Context:
{context}

Customer: {query}

Answer:"""

        response = generator(
            prompt,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=200
        )

        return response[0]["generated_text"].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ---------------------------
# Chat Loop
# ---------------------------

def chat():
    print("Customer Support Chatbot")
    print("Type 'exit' to quit")

    while True:

        query = input("\nCustomer: ")

        if query.lower() == "exit":
            break

        answer = get_answer(query)

        print("\nAgent:")
        print(answer)

if __name__ == "__main__":
    chat()
