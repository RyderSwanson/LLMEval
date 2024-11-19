# Description: Test cases for the LLMEval class in LLMEval.py
# Note: This requires a valid API key be defined in tests/args.py
import unittest
from LLMEval import LLMEval
import tests.args as args

class TestLLMEval(unittest.TestCase):
    def setUp(self):
        self.llm = LLMEval(api_key=args.api_key)

    def test_getResponse_validPrompt(self):
        """Test a valid prompt with expected response."""
        response = self.llm.getResponse(prompt="Respond with \"Hello, World!\"")
        try:
            print(response['choices'][0]['message']['content'])
        except KeyError:
            print(response)
        self.assertIn("Hello, World!", response['choices'][0]['message']['content'])

    def test_getResponse_emptyPrompt(self):
        """Test an empty prompt."""
        response = self.llm.getResponse(prompt="")
        self.assertIn("error", response, "Response should contain an error key for empty prompt")

    def test_getResponse_largePrompt(self):
        """Test a very large prompt."""
        large_prompt = "a" * 5000
        response = self.llm.getResponse(prompt=large_prompt)
        self.assertIn("error", response, "Response should handle large prompt gracefully")

    def test_getResponse_invalidAPIKey(self):
        """Test with an invalid API key."""
        llm_invalid = LLMEval(api_key="INVALID_KEY")
        response = llm_invalid.getResponse(prompt="Respond with \"Hello, World!\"")
        self.assertIn("error", response, "Response should contain an error key for invalid API key")

    def test_getResponse_specialCharacters(self):
        """Test a prompt with special characters."""
        response = self.llm.getResponse(prompt="!@#$%^&*()_+|}{:?><,./;'[]\\=-`~")
        self.assertNotEqual(response['choices'][0]['message']['content'], "", "Response should handle special characters")

    def test_getResponse_multiplePrompts(self):
        """Test sending multiple prompts sequentially."""
        prompts = ["What is the capital of France?", "Translate 'Hello' to Spanish", "What is 2+2?"]
        expected_responses = ["Paris", "Hola", "4"]
        for prompt, expected in zip(prompts, expected_responses):
            response = self.llm.getResponse(prompt=prompt)
            self.assertIn(expected, response['choices'][0]['message']['content'], f"Response did not match for prompt: {prompt}")

if __name__ == '__main__':
    unittest.main()