from google.genai import types
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=15000
    )
)

print("Testing Gemini API...")

start = time.time()

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with only: Gemini is working!"
    )

    print("\nSUCCESS!")
    print("Gemini response:", response.text)

except Exception as e:
    print("\nFAILED!")
    print("Error type:", type(e).__name__)
    print("Error details:", str(e))

finally:
    print("\nTime taken:", round(time.time() - start, 2), "seconds")


print("\nChecking available Gemini models...")

for model in client.models.list():
    if "flash" in model.name.lower():
        print(model.name)