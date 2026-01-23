#!/usr/bin/env python3
"""Simple multi-turn chatbot example following .github/prompts/chatbot-basic.prompt.md

This script demonstrates keeping conversation history and using the OpenAI Python
client to perform a short 3-question interaction about programming.
"""
import os
import sys
from openai import OpenAI


def get_client():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Export it and retry.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def run_chat():
    client = get_client()

    # System message sets assistant role/behavior
    messages = [
        {"role": "system", "content": "You are a helpful programming expert."}
    ]

    user_questions = [
        "What's a popular choice for a first programming language?",
        "What are some advantages of learning it as my first language?",
        "Can you show me a simple 'Hello World' program written in that language?",
    ]

    for q in user_questions:
        # Append user question to history
        messages.append({"role": "user", "content": q})

        # Call the chat completions endpoint (chat-based interaction)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.2,
        )

        # Extract assistant reply and append to history
        try:
            assistant_text = response.choices[0].message.content.strip()
        except Exception:
            assistant_text = response.choices[0].text.strip()

        messages.append({"role": "assistant", "content": assistant_text})

        # Print assistant response
        print("\nUser:", q)
        print("Assistant:", assistant_text)


def main():
    run_chat()


if __name__ == "__main__":
    main()
