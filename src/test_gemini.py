import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def test_gemini():
    """Test basic Gemini API call"""
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Make the API call
        prompt = "What is the best health insurance for freelancers in France?"
        response = model.generate_content(prompt)
        
        # Extract the response
        answer = response.text
        print("Gemini Response:")
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
    test_gemini()