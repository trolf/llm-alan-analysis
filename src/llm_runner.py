import os
import json
import time
import csv
from datetime import datetime
from dotenv import load_dotenv

# Import your working clients
from openai import OpenAI
import google.generativeai as genai
from mistralai.sdk import Mistral

load_dotenv()

class LLMRunner:
    def __init__(self):
        # Initialize all clients
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')  # Updated to requested model
        
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def _extract_openai_message_text(self, response):
        """Best-effort extraction of message text from OpenAI ChatCompletion response."""
        try:
            # Primary path for Chat Completions API
            content = getattr(response.choices[0].message, 'content', None)
            if content:
                return content
        except Exception:
            pass
        try:
            # If message is a dict-like
            message = getattr(response.choices[0], 'message', None)
            if isinstance(message, dict):
                content = message.get('content')
                if content:
                    return content
        except Exception:
            pass
        except Exception:
            pass
        # Fallback to stringified response for visibility
        return str(response)
    
    def ask_openai(self, prompt, run_number):
        """Ask OpenAI ChatGPT"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                # max_completion_tokens=500,
                # temperature not supported for this model; defaults to 1
            )
            # Robust extraction: capture actual content regardless of SDK representation
            message_content = self._extract_openai_message_text(response)
            return {
                "model": "gpt-5-mini",
                "prompt": prompt,
                "response": message_content,
                "run_number": run_number,
                "timestamp": datetime.now().isoformat(),
                "alan_mentioned": "alan" in message_content.lower(),
                "status": "success"
            }
        except Exception as e:
            return {"model": "gpt-5-mini", "prompt": prompt, "error": str(e), "status": "error", "run_number": run_number}
    
    def ask_gemini(self, prompt, run_number):
        """Ask Google Gemini"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return {
                "model": "gemini-2.5-flash",
                "prompt": prompt,
                "response": response.text,
                "run_number": run_number,
                "timestamp": datetime.now().isoformat(),
                "alan_mentioned": "alan" in response.text.lower(),
                "status": "success"
            }
        except Exception as e:
            return {"model": "gemini-2.5-flash", "prompt": prompt, "error": str(e), "status": "error", "run_number": run_number}
    
    def ask_mistral(self, prompt, run_number):
        """Ask Mistral"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.mistral_client.chat.complete(
                model="mistral-medium-2508",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            return {
                "model": "mistral-medium-2508",
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "run_number": run_number,
                "timestamp": datetime.now().isoformat(),
                "alan_mentioned": "alan" in response.choices[0].message.content.lower(),
                "status": "success"
            }
        except Exception as e:
            return {"model": "mistral-medium-2508", "prompt": prompt, "error": str(e), "status": "error", "run_number": run_number}
    
    def run_single_test(self, prompt, run_number=1):
        """Run a single prompt against all LLMs"""
        results = []
        
        # print(f"Testing prompt {run_number}: {prompt[:50]}...")
        
       # Test each LLM
        results.append(self.ask_openai(prompt, run_number))
        time.sleep(1)  # Rate limiting
        
        results.append(self.ask_gemini(prompt, run_number))
        time.sleep(1)
        
        results.append(self.ask_mistral(prompt, run_number))
        time.sleep(1)
        
        return results
    
    def run_all_tests(self, num_iterations=3):
        """Run all prompts against all LLMs multiple times"""
        # Load prompts
        with open('config/prompts.json', 'r') as f:
            prompts = json.load(f)
        
        all_results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n=== Prompt {i}/{len(prompts)} ===")
            
            for iteration in range(1, num_iterations + 1):
                print(f"Iteration {iteration}/{num_iterations}")
                results = self.run_single_test(prompt, iteration)
                all_results.extend(results)
        
        return all_results
    
    def save_to_csv(self, results, filename="results.csv"):
        """Save results to CSV"""
        if not results:
            return
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        filepath = f"data/{filename}"
        
        fieldnames = ['model', 'prompt', 'alan_mentioned', 'response', 'run_number', 'timestamp',  'status', 'error']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Fill in missing fields
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"\nResults saved to {filepath}")

# Test script
if __name__ == "__main__":
    runner = LLMRunner()
    
    # Run all prompts for multiple iterations
    with open('config/prompts.json', 'r') as f:
        prompts = json.load(f)
    
    print(f"Running full test on {len(prompts)} prompts...")
    results = runner.run_all_tests(num_iterations=2)
    
    # Show a brief summary of results
    for result in results:
        if result['status'] == 'success':
            print(f"\n{result['model']} | run {result['run_number']}: Alan mentioned = {result['alan_mentioned']}")
            print(f"Response: {result['response'][:100]}...")
        else:
            print(f"\n{result['model']} | run {result['run_number']}: ERROR - {result.get('error', 'Unknown error')}")
    
    # Save aggregated results
    runner.save_to_csv(results, "test_results.csv")