import os
import LLMEval as args

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
cmd = "python -m LLMEval prompt --api_key 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"
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)

input("Press Enter to continue...")
cmd = "python -m LLMEval prompt --api_key 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 --model meta-llama/llama-3.1-8b-instruct:free"
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)

input("Press Enter to continue...")

cmd = "python -m LLMEval prompt --api_key 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 --model meta-llama/llama-3.1-8b-instruct:free --prompt \"What are you?\""
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)

input("Press Enter to continue...")
cmd = "python -m LLMEval test"
print(f"Running following command: {cmd}\n")

input("Press Enter to continue...")
os.system(cmd)