import os
import tests.args as args

cmd = "python -m LLMEval"
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")

os.system(cmd)

input("Press Enter to continue...")

cmd = "python -m LLMEval prompt"
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)

input("Press Enter to continue...")
cmd = f"python -m LLMEval prompt --api_key {args.api_key}"
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)

input("Press Enter to continue...")
cmd = f"python -m LLMEval prompt --api_key {args.api_key} --model meta-llama/llama-3.1-8b-instruct:free"
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)

input("Press Enter to continue...")

cmd = f"python -m LLMEval prompt --api_key {args.api_key} --model meta-llama/llama-3.1-8b-instruct:free --prompt \"What are you?\""
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)

input("Press Enter to continue...")
cmd = "python -m LLMEval test"
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)