import os
from google import genai
from google.genai import types

# Step 4 & 5: Load the API key securely from environment variables.
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY environment variable not found.")
    print("Please set it in your terminal before running the script.")
    exit()

# Initialize the NEW client with your API key
client = genai.Client(api_key=api_key)

print("🤖 Welcome to your Command-Line AI Assistant!")
print("Type 'exit' or 'quit' to end the conversation.\n")

# Practical Task: Command-line loop
while True:
    # User input
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    if not user_input.strip():
        continue

    try:
        print("AI is thinking...")
        
        # Step 7, 8 & 9: Make the request using the new syntax and model settings
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=user_input,
            config=types.GenerateContentConfig(
                temperature=0.7,           # Controls randomness/creativity
                max_output_tokens=500      # Limits response length
            )
        )

        # Step 10: Output the API response to the user
        print(f"🤖 AI: {response.text}\n")
        
    except Exception as e:
        print(f"An error occurred: {e}\n")