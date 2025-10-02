import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_perplexity():
    """Test basic Perplexity API call with correct model name"""
    try:
        url = "https://api.perplexity.ai/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar",  # Updated model name!
            "messages": [
                {
                    "role": "user",
                    "content": "What is the best health insurance for freelancers in France?"
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            
            print("Perplexity Response:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            
            # Check if Alan is mentioned
            alan_mentioned = "alan" in answer.lower()
            print(f"Alan mentioned: {alan_mentioned}")
            
            return answer
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_perplexity()