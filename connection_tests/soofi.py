import os
from openai import OpenAI
import streamlit as st

# client = OpenAI(
#     base_url="https://soofi-owu.l3s.de/api/chat/completions",
#     api_key="sk-ed443de23ce34d09b86ca3ef4df325f4",
# )

# # Example usage
# response = client.chat.completions.create(
#     model="sft_Soofi_Nano_30B_A3B_nemotron_posttrain_v3_em_v2_cleaned_bridge__iter_0000600",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Wie heißt die Hauptstadt Niedersachsens?"}
#     ]
# )

import json
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://soofi-owu.l3s.de/api/chat/completions"

headers = {
    "Authorization": "Bearer sk-ed443de23ce34d09b86ca3ef4df325f4",
    "Content-Type": "application/json",
}

payload = {
    "model": "sft_Soofi_Nano_30B_A3B_nemotron_posttrain_v3_em_v2_cleaned_bridge__iter_0000600",
    "messages": [
        {"role": "user", "content": "Wie heißt die Hauptstadt von Niedersachsen?"}
    ],
    "temperature": 0.2,
    "max_tokens": 256,
    "chat_template_kwargs": {
        "enable_thinking": False
    },
    "stream": True,
}

with requests.post(url, headers=headers, json=payload, verify=False, stream=True) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")
        if content:
            print(content, end="", flush=True)
    print()  # trailing newline