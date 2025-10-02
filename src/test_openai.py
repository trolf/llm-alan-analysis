import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def test_chatgpt():
    """Test basic ChatGPT API call"""
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the best health insurance for freelancers in France?"}
            ],
           # max_completion_tokens=1000
        )
        
        # Extract the response
        answer = response.choices[0].message.content
        print("ChatGPT Response:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_chatgpt()