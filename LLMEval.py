import requests
import json
import argparse

class LLMEval:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def getResponse(self, prompt: str, model: str = "openai/gpt-3.5-turbo"):

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                # "HTTP-Referer": f"{YOUR_SITE_URL}", # Optional, for including your app on openrouter.ai rankings.
                # "X-Title": f"{YOUR_APP_NAME}", # Optional. Shows in rankings on openrouter.ai.
            },
            data=json.dumps({
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            })
        )

        return response.json()

# TODO: This is for testing. Remove this when done testing.
if __name__ == "__main__":
    import tests.args as args
    from Models import Models
    llm = LLMEval(api_key=args.api_key)
    models = Models(api_key=args.api_key)
    model = models.getRandomFreeModelID()
    response = llm.getResponse(prompt="What is the meaning of life?", model=model)
    print(response)
    print(model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool to measure the performance of LLMs')
    parser.add_argument('--api_key', type=str, help='API Key', required=True)
    parser.add_argument('--model', type=str, default="openai/gpt-3.5-turbo", help='Model')
    parser.add_argument('--prompt', type=str, default="What is the meaning of life?", help='Prompt')
    args = parser.parse_args()

    llm = LLMEval(api_key=args.api_key)
    response = llm.getResponse(prompt=args.prompt)
    print(response)