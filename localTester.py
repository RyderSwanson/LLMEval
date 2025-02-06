from EvaluationHandler import evaluatorHandler


def tester():

    myApiKey = "Your API Key"
    handlerClass = evaluatorHandler(myApiKey)
    handlerClass.handleEvaluations()

tester()