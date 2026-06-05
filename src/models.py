import os
import google.generativeai as genai

# Make sure you set your API key, or pass it directly here
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

print("Available Models for Generation:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)