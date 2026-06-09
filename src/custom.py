import os
from google import genai


os.environ["GEMINI_API_KEY"] = ""  # replace with your Gemini API key

# Initialize the standard Google GenAI client
client = genai.Client()

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