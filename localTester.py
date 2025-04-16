from EvaluationHandler import evaluatorHandler


def tester():

    myApiKey = "APIKey" #Enter your api key
    handlerClass = evaluatorHandler(myApiKey, "ModelName") #Enter your model name
    handlerClass.handleEvaluations()

tester()