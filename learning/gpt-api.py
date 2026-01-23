from openai import OpenAI

API_KEY = "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"

client = OpenAI(
  api_key=API_KEY,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant that translates English to Swahili."},
    {"role": "user", "content": "Translate the following English text to Swahili: 'Hello, how are you?'"}
]
model = "gpt-3.5-turbo"

completion = client.chat.completions.create(
  model=model,
  messages=messages
)

print(completion.choices[0].message.content)