import os
from openai import OpenAI
import streamlit as st

client = OpenAI(
    base_url="https://chat-ai.academiccloud.de/v1/",
    api_key=st.secrets["KISSKI_API_KEY"],
)

# List available models
models = client.models.list()
print('Available chat/completion models:')
for model in models:
    print(model.id)

# Example usage
response = client.chat.completions.create(
    model="qwen3.5-27b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

# # Try to get embedding models (this may not work with all providers)
# print('\nTrying to list embedding models...')
# try:
#     # Some providers expose embedding models through the models endpoint
#     embedding_models = [model for model in models if 'embedding' in model.id.lower()]
#     if embedding_models:
#         print('Embedding models found in models list:')
#         for model in embedding_models:
#             print(model.id)
#     else:
#         print('No embedding models found in models list')
# except Exception as e:
#     print(f'Error listing embedding models: {e}')

# # Alternative approach: Try common embedding model names
# print('\nTrying common embedding model names...')
# common_embedding_models = [
#     "qwen3-embedding-4b",
#     "text-embedding-ada-002", 
#     "text-embedding-3-small",
#     "text-embedding-3-large",
#     "embedding-001",
#     "text-embedding-002"
# ]

# working_models = []
# for model_name in common_embedding_models:
#     try:
#         test_embedding = client.embeddings.create(
#             input=["test"], 
#             model=model_name
#         )
#         working_models.append(model_name)
#         print(f"✓ {model_name} - Working")
#     except Exception as e:
#         print(f"✗ {model_name} - Not available: {str(e)[:50]}...")

# if working_models:
#     print(f'\nWorking embedding models: {working_models}')
# else:
#     print('\nNo common embedding models found')

# # Test the original embedding
# print('\nTesting original embedding...')
# embeddings = client.embeddings.create(input=["My name jeff"], model="qwen3-embedding-4b").data[0].embedding
# print(f"Embedding vector length: {len(embeddings)}")
# print(f"First 5 values: {embeddings[:5]}")