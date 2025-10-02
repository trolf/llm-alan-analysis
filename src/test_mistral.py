import os
from dotenv import load_dotenv
from mistralai.sdk import Mistral
import time

# Load environment variables
load_dotenv()

# Initialize Mistral client (new SDK)
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def test_mistral():
    """Test basic Mistral API call"""
    try:
        # Create the message (dict format supported by the SDK)
        messages = [
            {"role": "user", "content": "What is the best health insurance for freelancers in France?"}
        ]
        
        # Make the API call
        response = client.chat.complete(
            model="mistral-medium-2508",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract the response
        answer = response.choices[0].message.content
        print("Mistral Response:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        
        # Check if Alan is mentioned
        alan_mentioned = "alan" in answer.lower()
        print(f"Alan mentioned: {alan_mentioned}")
        
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_mistral()