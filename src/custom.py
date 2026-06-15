import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing Gemini API key. Add GEMINI_API_KEY to src/.env")

client = genai.Client(api_key=api_key)

print("Fetching available models...\n")

# Call the list models service
for model in client.models.list():
    print(f"Model Name: {model.name}")
    
    # Safely try to print the description if it exists
    if hasattr(model, 'description') and model.description:
        print(f"Description: {model.description}")
        
    # Highlight embedding models so you don't have to squint
    if "embed" in model.name.lower():
        print("🟢 THIS IS AN EMBEDDING MODEL!")
        
    print("-" * 40)