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
    model="qwen3.6-35b-a3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print("Chat test successful. Response:")
print(response.choices[0].message.content)

embeddings = client.embeddings.create(input=["My name jeff"], model="qwen3-embedding-4b").data[0].embedding
print(f"Embedding test successful")
print(f"Embedding vector length: {len(embeddings)}")


import streamlit as st
from openai import OpenAI

def check_safety_llama_guard_3(text):
    """
    Checks the safety of the given text using the LlamaGuard-3 model.
    """
    try:
        client = OpenAI(
            base_url=st.secrets["daisec_url"],
            api_key=st.secrets["DAISEC_API_KEY"]
        )

        response = client.chat.completions.create(
            model=st.secrets["safety_model"],
            messages=[{"role": "user", "content": text}],
            max_tokens=32000
        )

        safety_rating = response.choices[0].message.content
        print(f"safety_rating: {safety_rating}")

        if safety_rating == "safe":
            return True, None
        else:
            # Otherwise: Unsafe content detected, model outputs "Unsafe\n<Unsafe content categories>"
            # E.g., Unsafe\nS1,S2,S3
            categories = safety_rating.split("\n")[1].split(",")

            category_explanation = {
                "S1": "Gewaltverbrechen.",
                "S2": "Nicht-gewalttätige Verbrechen.",
                "S3": "Sexuelle Straftaten.",
                "S4": "Kinderausbeutung.",
                "S5": "Beleidigung und Verleumdung.",
                "S6": "Spezielle Beratung (z.B. medizinisch oder rechtlich).",
                "S7": "Datenschutz und Privatsphäre.",
                "S8": "Urheberrecht und geistiges Eigentum.",
                "S9": "Waffen und gefährliche Gegenstände.",
                "S10": "Hass und Diskriminierung.",
                "S11": "Selbstverletzung und Suizid.",
                "S12": "Sexuelle Inhalte.",
                "S13": "Wahlen und Politik.",
            }

            explanation = "Leider enthält deine Nachricht gewisse Inhalte, die als unsicher eingestuft wurden. Diese Inhalte sind:\n"
            for category in categories:
                explanation += f"- {category_explanation[category]}\n"
            explanation += "Bitte überprüfe deine Nachricht und versuche es erneut."

            return False, explanation
    except Exception as e:
        print(f"Error checking safety: {e}")
        return True, None

check_safety_llama_guard_3("This is a test message to check safety.")

print("Safety check completed.")

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