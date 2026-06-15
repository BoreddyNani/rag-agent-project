import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing Gemini API key. Add GEMINI_API_KEY to src/.env")

# Make sure you set your API key, or pass it directly here
genai.configure(api_key=api_key)

print("Available Models for Generation:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)