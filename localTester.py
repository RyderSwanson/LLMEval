from EvaluationHandler import evaluatorHandler


def tester():


    handlerClass = evaluatorHandler(myApiKey)
    handlerClass.handleEvaluations()
    print("Didnt Fail!")

tester()