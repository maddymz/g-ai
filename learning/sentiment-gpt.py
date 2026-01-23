from openai import OpenAI
import os
import sys

# Load API key from environment for security. Export with:
#   export OPENAI_API_KEY="sk-..."
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set. Please export it and retry.", file=sys.stderr)
        sys.exit(1)

client = OpenAI(api_key=API_KEY)

TEXT_TO_ANALYZE = "It's a confusing process, and the instructions provided don't help at all."

def analyze_sentiment(text):
    """Analyze sentiment of the given text."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Analyze the sentiment of the following text: \"{text}\". Is it positive, negative, or neutral? Answer in one word with no punctuation."}
        ],
        max_tokens=50
    )

    sentiment = response.choices[0].message.content.strip().lower()

    # Validate sentiment to ensure it's one of the three categories
    if sentiment not in ["positive", "negative", "neutral"]:
        return "Unable to determine sentiment. Please try again."
    
    return f"The sentiment of the text is: {sentiment}"


def main():
    result = analyze_sentiment(TEXT_TO_ANALYZE)
    print(result)

if __name__ == "__main__":
    main()

