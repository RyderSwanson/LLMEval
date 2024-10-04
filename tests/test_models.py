# Description: This file tests the Models class in models.py
from Models import Models
import unittest

class TestModels(unittest.TestCase):
    def setUp(self):
        self.models = Models()

    def test_getModels(self):
        response = self.models.getModels()
        self.assertIn('data', response)

if __name__ == '__main__':
    unittest.main()