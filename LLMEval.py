import requests
import json
import argparse
import time
from Models import Models

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

        id = response.json()['id']
        print(f"https://openrouter.ai/api/v1/generation?id={id}")

        # TODO: Currently we are just hoping the stats are avaliable in 1 second. We should wait until they are avaliable.
        time.sleep(1)
        stats = requests.request("GET", f"https://openrouter.ai/api/v1/generation?id={id}", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
            })

        #combine into one json
        response = response.json()
        response['stats'] = stats.json()

        return response

if __name__ == "__main__":
    models = Models()
    parser = argparse.ArgumentParser(description='Tool to measure the performance of LLMs')
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for the prompt command
    prompt_parser = subparsers.add_parser('prompt', help='Send a prompt to the model')
    prompt_parser.add_argument('--api_key', type=str, help='API Key', required=True)
    prompt_parser.add_argument('--model', type=str, default=models.getRandomFreeModelID(), help='Model')
    prompt_parser.add_argument('--prompt', type=str, default="What is the meaning of life?", help='Prompt')
    prompt_parser.add_argument('--temperature', type=float, default=1.0, help='Temperature')
    prompt_parser.add_argument('--top_p', type=float, default=1.0, help='Top p')
    prompt_parser.add_argument('--top_k', type=int, default=0, help='Top k')
    prompt_parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty')
    prompt_parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty')
    prompt_parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')
    prompt_parser.add_argument('--min_p', type=float, default=0.0, help='Min p')
    prompt_parser.add_argument('--top_a', type=float, default=0.0, help='Top a')
    prompt_parser.add_argument('--seed', type=int, help='Seed')
    prompt_parser.add_argument('--max_tokens', type=int, help='Max tokens')

    # Subparser for the test command
    test_parser = subparsers.add_parser('test', help='Run a predefined test')
    test_parser.add_argument('--api_key', type=str, help='API Key', required=True)
    test_parser.add_argument('--test', type=str, help='Test to run', required=True)

    args = parser.parse_args()

    # If no command is provided, print help and exit
    if not args.command:
        parser.print_help()
        print("\nExample usage:")
        print("  python -m LLMEval prompt --api_key YOUR_API_KEY --prompt 'What is the meaning of life?'")
        print("  python -m LLMEval test --api_key YOUR_API_KEY --test 'test_name'")
        exit(1)

    print(args)

    llm = LLMEval(api_key=args.api_key)

    if args.command == 'prompt':
        response = llm.getResponse(
            prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
            top_a=args.top_a,
            seed=args.seed,
            max_tokens=args.max_tokens
        )

        print(f"Model used: {args.model}\n")
        print(f"Prompt: {args.prompt}\n")
        print(f"Model response: {response['choices'][0]['message']['content']}")

        # Save the response to a json file
        with open(f"responses/{args.model.replace('/', '_').replace('.', '_').replace(':', '_')}_{int(time.time())}.json", "w") as f:
            json.dump(response, f, indent=4)


    elif args.command == 'test':
        # Placeholder for test execution logic
        print(f"Running test: {args.test}")
        # Implement the test logic here
