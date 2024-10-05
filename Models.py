# Description: This file can query openrouter for the all the available models.

import requests
import random
import copy
import tests.args as args

class Models:
    def __init__(self):
        pass

    def getModels(self):
        response = requests.get(
            url="https://openrouter.ai/api/v1/models"
        )
        return response.json()
    
    def getFreeModels(self):
        response = self.getModels()
        free_models = []
        for model in response['data']:
            prices = copy.copy(model['pricing'])
            for x in prices:
                prices[x] = float(prices[x])
            if sum(prices.values()) == 0:
                free_models.append(model)
        return free_models
    
    def getRandomModel(self):
        response = self.getModels()
        response = list(response['data'])
        return response[random.randint(0, len(response) - 1)]
    
    def getRandomModelID(self):
        response = self.getModels()
        response = list(response['data'])
        return response[random.randint(0, len(response) - 1)]['id']

    def getRandomFreeModel(self):
        response = self.getFreeModels()
        return response[random.randint(0, len(response) - 1)]
    
    def getRandomFreeModelID(self):
        response = self.getFreeModels()
        return response[random.randint(0, len(response) - 1)]['id']

# TODO: This is just for testing purposes. Remove this when done testing.
if __name__ == "__main__":
    models = Models()
    
    print(models.getRandomModelID())