import os
import time

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


client = genai.Client(api_key=api_key)


MODEL = "gemini-3.5-flash-lite"


for attempt in range(3):

    try:

        print(f"Attempt {attempt + 1}...")

        response = client.models.generate_content(
            model=MODEL,
            contents="Explain database normalization in two simple sentences."
        )

        print("\nGemini response:\n")
        print(response.text)

        break

    except Exception as e:

        print(f"Request failed: {e}")

        if attempt < 2:
            print("Retrying in 5 seconds...\n")
            time.sleep(5)

        else:
            print("\nGemini request failed after 3 attempts.")