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
        """Ask OpenAI via Responses API with web search and citation extraction"""
        try:
            response = self.openai_client.responses.create(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": "You are a helpful assistant. Cite your sources when possible."},
                    {"role": "user", "content": prompt}
                ],
                tools=[
                    {"type": "web_search"}
                ],
                # max_output_tokens=500,
            )

            # Extract combined text answer
            message_content = getattr(response, "output_text", None)
            if not message_content:
                fragments = []
                try:
                    for item in getattr(response, "output", []) or []:
                        for content in getattr(item, "content", []) or []:
                            if getattr(content, "type", None) == "output_text" and getattr(content, "text", None):
                                value = getattr(content.text, "value", None) or getattr(content.text, "content", None)
                                if value:
                                    fragments.append(value)
                except Exception:
                    pass
                message_content = "\n".join(fragments) if fragments else ""

            # Collect citations from annotations if present
            citations = []
            try:
                seen = set()
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", []) or []:
                        if getattr(content, "type", None) == "output_text" and getattr(content, "text", None):
                            annotations = getattr(content.text, "annotations", None) or []
                            for ann in annotations:
                                ann_type = getattr(ann, "type", None)
                                if ann_type == "web_citation":
                                    url = getattr(ann, "url", None)
                                    title = getattr(ann, "title", None)
                                    key = (url, title)
                                    if url and key not in seen:
                                        citations.append({"type": "web", "url": url, "title": title})
                                        seen.add(key)
                                elif ann_type == "file_citation":
                                    file_id = getattr(getattr(ann, "file_citation", None), "file_id", None)
                                    quote = getattr(getattr(ann, "file_citation", None), "quote", None)
                                    key = (file_id, quote)
                                    if file_id and key not in seen:
                                        citations.append({"type": "file", "file_id": file_id, "quote": quote})
                                        seen.add(key)
            except Exception:
                pass

            # Append citations as a Markdown table after the raw answer (schema unchanged)
            if citations:
                lines = [message_content, "", "Citations:", "", "| # | Type | Title/Quote | URL / File ID |", "|---:|------|-------------|----------------|"]
                for idx, c in enumerate(citations, start=1):
                    if c["type"] == "web":
                        title_or_quote = c.get("title") or ""
                        ref = c.get("url") or ""
                        lines.append(f"| {idx} | web | {title_or_quote} | {ref} |")
                    elif c["type"] == "file":
                        title_or_quote = c.get("quote") or ""
                        ref = c.get("file_id") or ""
                        lines.append(f"| {idx} | file | {title_or_quote} | {ref} |")
                message_content = "\n".join(lines)

            return {
                "model": "gpt-5-mini",
                "prompt": prompt,
                "response": message_content,
                "run_number": run_number,
                "timestamp": datetime.now().isoformat(),
                "alan_mentioned": "alan" in (message_content or "").lower(),
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
    
    def run_all_tests(self, num_iterations=1):
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
    results = runner.run_all_tests(num_iterations=1)
    
    # Show a brief summary of results
    for result in results:
        if result['status'] == 'success':
            print(f"\n{result['model']} | run {result['run_number']}: Alan mentioned = {result['alan_mentioned']}")
            print("Response:")
            print(result['response'])
        else:
            print(f"\n{result['model']} | run {result['run_number']}: ERROR - {result.get('error', 'Unknown error')}")
    
    # Save aggregated results
    runner.save_to_csv(results, "test_results.csv")