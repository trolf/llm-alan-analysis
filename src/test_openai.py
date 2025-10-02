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
    """Test Responses API call with citations extraction"""
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": "You are a helpful assistant. Cite your sources when possible."},
                {"role": "user", "content": "What is the best health insurance for freelancers in France?"}
            ],
            tools=[
                {"type": "web_search"}
            ],
            # max_output_tokens=1000  # Optional equivalent for responses API
        )

        # Extract combined text answer
        answer = getattr(response, "output_text", None)
        if not answer:
            # Fallback: concatenate any text fragments from the output
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
            answer = "\n".join(fragments) if fragments else ""

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
                            # Web citation shape
                            if ann_type == "web_citation":
                                url = getattr(ann, "url", None)
                                title = getattr(ann, "title", None)
                                key = (url, title)
                                if url and key not in seen:
                                    citations.append({"type": "web", "url": url, "title": title})
                                    seen.add(key)
                            # File citation shape
                            elif ann_type == "file_citation":
                                file_id = getattr(getattr(ann, "file_citation", None), "file_id", None)
                                quote = getattr(getattr(ann, "file_citation", None), "quote", None)
                                key = (file_id, quote)
                                if file_id and key not in seen:
                                    citations.append({"type": "file", "file_id": file_id, "quote": quote})
                                    seen.add(key)
        except Exception:
            pass

        print("ChatGPT Response:")
        print("-" * 50)
        print(answer or "<no text output>")
        print("-" * 50)

        if citations:
            print("Citations:")
            for c in citations:
                if c["type"] == "web":
                    if c.get("title"):
                        print(f"- {c['title']}: {c['url']}")
                    else:
                        print(f"- {c['url']}")
                elif c["type"] == "file":
                    if c.get("quote"):
                        print(f"- file_id={c['file_id']} — \"{c['quote']}\"")
                    else:
                        print(f"- file_id={c['file_id']}")

        return answer

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_chatgpt()