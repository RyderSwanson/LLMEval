import os
import sys
from EvaluationModules import EvaluationMethodFactory, getEvaluators
import multiprocessing
from LLMEval import LLMEval

class evaluatorHandler():

    def __init__(self, key):

        self.apiKey = key
        self.llm = LLMEval(api_key=key)
        #This section will be used to add required dependencies to the current location, although it has not yet been made
        self.isInBetaMode = False

    def handleEvaluations(self):

        listOfEvaluators = getEvaluators()
        listOfEvaluationModules = []
        listOfResponses = []

        print(listOfEvaluators)

        for evaluator in listOfEvaluators:

            response = ""
            self.prompt = None
            self.prompt = self.getLLMPrompt(evaluator)

            if(self.prompt != None):

                if(self.isInBetaMode == True):

                    response = self.getEvalutorResponse(evaluator)

                else:

                    response = self.llm.getResponse(prompt=self.prompt)['choices'][0]['message']['content']

            else:
                
                raise Exception("Critical Failure: No Prompt Found For: " + str(evaluator))

            if len(response) != 0:

                evaluationDataList = self.getEvalutorData(evaluator)
                evaluationDataList.insert(0, response)
                evaluationModule = EvaluationMethodFactory.create_evaluation(evaluator, evaluationDataList)
                listOfEvaluationModules.append(evaluationModule)

            else: 

                raise Exception("Critical Failure: No Response From LLM Found")

        # with multiprocessing.Pool(processes=len(listOfEvaluationModules)) as pool:
        #     # Map the evaluate_instance function to each class instance
        #     results = pool.map(self.callEvaluator, listOfEvaluationModules)
        #     print(results)

        results = []

        for process in listOfEvaluationModules:
            results.append(process.PerformEvaluation())

        print(results)
        print("Need to call backend")
        #callBackend(listOfEvaluators, results)

    def callEvaluator(instance):

        return instance.PerformEvaluation()

    def getEvalutorData(self, evaluatorName):
        
        data = None

        for file in os.listdir("evaluatorData/"):

            if evaluatorName in file: 

                with open(os.path.join("evaluatorData", file), 'r') as f:

                    data = f.readlines()

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

                            response = f.readlines()[0]
                            break
                
        return response
    
    def getEvalutorResponse(self, evaluatorName):
        
        data = None

        for file in os.listdir("evaluatorResponses/"):

            if evaluatorName in file: 

                with open(os.path.join("evaluatorResponses", file), 'r') as f:

                    data = f.readlines()

        return data[0]