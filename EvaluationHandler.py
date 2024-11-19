import os
import sys
import EvaluationModules as evalModules
import multiprocessing
from LLMEval import LLMEval

class evaluatorHandler():

    def __init__(self, key):

        self.apiKey = key
        self.llm = LLMEval(api_key=key)
        #This section will be used to add required dependencies to the current location, although it has not yet been made
        self.isInBetaMode = False

    def handleEvaluations(self):

        listOfEvaluators = []
        listOfResponses = []

        for evaluator in evalModules.getEvaluators():

            self.prompt = self.getLLMPrompt(evaluator)

            if(self.prompt != None):

                if(self.isInBetaMode == True):

                    listOfResponses.append(self.getEvalutorData(evaluator))

                else:

                    print("Inside of caller")
                    print("")
                    listOfResponses.append(self.llm.getResponse(prompt=self.prompt))
                    input()
            else:
                
                raise Exception("Critical Failure: No Prompt Found For: " + str(evaluator))

            if len(listOfResponses) != 0:

                listOfEvaluators.append(evalModules.EvaluationMethodFactory.create_evaluation(evaluator, listOfResponses))

            else: 

                raise Exception("Critical Failure: No Response From LLM Found")

        with multiprocessing.Pool(processes=len(listOfEvaluators)) as pool:
            # Map the evaluate_instance function to each class instance
            results = pool.map(self.callEvaluator, listOfEvaluators)

        print("Need to call backend")
        #callBackend(listOfEvaluators, results)

    def callEvaluator(instance):

        return instance.PerformEvaluation()

    def getEvalutorData(self, evaluatorName):
        
        data = None

        for file in os.path.dirname("evaluatorData/"):

            if evaluatorName in file: 

                with open(file, 'r') as f:

                    data = f.readlines()
        
        for file in os.path.dirname("evaluatorPrompt/"):

            if evaluatorName in file: 

                with open(file, 'r') as f:

                    if data != None:

                        data = (f.readline(),data)
                    else:

                        data = f.readline()

        return data
    
    """
        Author: Elliott Hendrickson
        Input: Evaluator Name
        Output: Prompt
    """
    def getLLMPrompt(self, evaluatorName):

        response = None

        for dirpath, dirnames, filenames in os.walk("evaluatorPrompt"):

            for filename in filenames:

                if evaluatorName in filename: 

                    with open(os.path.join(dirpath, filename), 'r') as f:

                        if response == None:

                            response = f.readlines()
                            break
                
        return response