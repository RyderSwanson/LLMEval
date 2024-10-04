# Description: Test cases for the LLMEval class in LLMEval.py
# Note: This requries a valid API key be defined in tests/args.py
import unittest
from LLMEval import LLMEval
import tests.args as args

class TestLLMEval(unittest.TestCase):
    def setUp(self):
        self.llm = LLMEval(api_key=args.api_key)

    def test_getResponse(self):
        response = self.llm.getResponse(prompt="Respond with \"Hello, World!\"")
        print(response['choices'][0]['message']['content'])
        self.assertIn( "Hello, World!", response['choices'][0]['message']['content'])

if __name__ == '__main__':
    unittest.main()