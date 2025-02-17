from EvaluationHandler import evaluatorHandler


def tester():

    myApiKey = "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"
    handlerClass = evaluatorHandler(myApiKey)
    handlerClass.handleEvaluations()

tester()