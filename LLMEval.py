import requests
import json
import argparse

class LLMEval:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def getResponse(self, 
                    prompt: str, 
                    model: str = "openai/gpt-3.5-turbo",
                    temperature: float = 1.0,
                    top_p: float = 1.0,
                    top_k: int = 0,
                    frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0,
                    repetition_penalty: float = 1.0,
                    min_p: float = 0.0,
                    top_a: float = 0.0,
                    seed: int = None,
                    max_tokens: int = None):
        """This function takes a prompt and optional parameters, such as model, and returns a json response from the model.

        Args:
            prompt (str): The prompt to send to the model.
            model (str, optional): The model to use. Defaults to "openai/gpt-3.5-turbo".
            temperature (float, optional): The temperature to use. Defaults to 1.0, can be 0.0 to 2.0.
            top_p (float, optional): The top_p to use. Defaults to 1.0, can be 0.0 to 1.0.
            top_k (int, optional): The top_k to use. Defaults to 0, can be 0 or above.
            frequency_penalty (float, optional): The frequency_penalty to use. Defaults to 0.0, can be -2.0 to 2.0.
            presence_penalty (float, optional): The presence_penalty to use. Defaults to 0.0, can be -2.0 to 2.0.
            repetition_penalty (float, optional): The repetition_penalty to use. Defaults to 1.0, can be 0.0 to 2.0.
            min_p (float, optional): The min_p to use. Defaults to 0.0, can be 0.0 to 1.0.
            top_a (float, optional): The top_a to use. Defaults to 0.0, can be 0.0 to 1.0.
            seed (int, optional): The seed to use. Defaults to None.
            max_tokens (int, optional): The max_tokens to use. Defaults to None.

        Returns:
            response.json(): The json response from the model.

        Note:
            For full details on all the parameters, visit https://openrouter.ai/docs/parameters
        
        """

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
                ],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repetition_penalty": repetition_penalty,
                "min_p": min_p,
                "top_a": top_a,
                "seed": seed,
                "max_tokens": max_tokens
            })
        )

        return response.json()

# TODO: This is for testing. Remove this when done testing.
if __name__ == "__main__":
    import tests.args as args
    from Models import Models
    llm = LLMEval(api_key=args.api_key)
    models = Models()
    model = models.getRandomFreeModelID()
    response = llm.getResponse(prompt="What is the meaning of life?", model=model)
    print(response)
    print(model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool to measure the performance of LLMs')
    parser.add_argument('--api_key', type=str, help='API Key', required=True)
    parser.add_argument('--model', type=str, default=models.getRandomFreeModelID(), help='Model')
    parser.add_argument('--prompt', type=str, default="What is the meaning of life?", help='Prompt')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top p')
    parser.add_argument('--top_k', type=int, default=0, help='Top k')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')
    parser.add_argument('--min_p', type=float, default=0.0, help='Min p')
    parser.add_argument('--top_a', type=float, default=0.0, help='Top a')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--max_tokens', type=int, help='Max tokens')

    args = parser.parse_args()

    llm = LLMEval(api_key=args.api_key)
    response = llm.getResponse(prompt=args.prompt, model=args.model)

    print(f"Model used: {args.model}\n")
    print(f"Prompt: {args.prompt}\n")
    print(f"Model response: {response['choices'][0]['message']['content']}")