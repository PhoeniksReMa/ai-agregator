import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class OpenAiAPIServise:
    def __init__(self):
        self.apikey = OPENAI_API_KEY

    def chat_complete(self, model, messages, **settings):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **settings
        )
        completion = response.choices[0].message
        return completion