import os
import sys
import EvaluationModules as evalModules

class evaluatorHandler():

    def __init__(self):
        
        #This section will be used to add required dependencies to the current location, although it has not yet been made
        print("This section will be used to add required dependencies to the current location, although it has not yet been made")
    
    def handleEvaluations(self):

        listOfEvaluators = []

        for evaluator in evalModules.getEvaluators():

            data = self.getEvalutorData(evaluator)
            listOfEvaluators.append(evalModules.EvaluationMethodFactory.create_evaluation(evaluator, ))

    def getEvalutorData(self, evaluatorName):
        
        outputList = []

        for file in os.path.dirname("evaluatorData/"):

            with open(file, 'r') as f:

                outputList.append((file, f.readlines()))
                
        return outputList

evaluatorHandler().handleEvaluations()