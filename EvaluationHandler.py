import os
import sys
from EvaluationModules import conductEvaluation
import multiprocessing
from LLMEval import LLMEval
import pandas as pd
from datetime import date
import numpy as np

class evaluatorHandler():

    def __init__(self, key):

        self.apiKey = key
        self.llm = LLMEval(api_key=key)
        #This section will be used to add required dependencies to the current location, although it has not yet been made
        self.isInBetaMode = False

    def handleEvaluations(self):

        nonAggregateResults = pd.DataFrame()
        aggregateResults = pd.DataFrame()

        for index, row in self.getPrompts().iterrows():

            print("Working on prompt: " + str(index))
            fullResponse = self.llm.getResponse(prompt=row["prompt"])
            response = fullResponse['choices'][0]['message']['content']
            modelName = fullResponse['model']
            # print("Type: " + row["type"])

            nonAggregateResults = pd.concat([nonAggregateResults, conductEvaluation(row, response)], ignore_index=True)

        # print(nonAggregateResults)
        results = self.aggregateData(nonAggregateResults)
        # results = nonAggregateResults.apply(pd.to_numeric, errors='coerce').mean(axis=1)
        # results = nonAggregateResults.apply(self.custom_mean)

        results['CurrentDate'] = date.today()
        results['modelName'] = modelName
        # oldResults = pd.read_csv('path_to_file.csv')
        # newOutput = pd.concat([oldResults], ignore_index=True, axis = 0)
        # newOutput['results'] = np.nan
        # newOutput['results'] = newOutput.mean(axis=1)
        if os.path.exists("results.csv"):
            previousData = pd.read_csv("results.csv")
            pd.concat((previousData, results), axis = 0).to_csv("results.csv", index=False)

        else:
            results.to_csv("results.csv", index=False)
        # #callBackend(listOfEvaluators, results)

    def aggregateData(self, dataframe):

        outputDataFrame = pd.DataFrame()

        for col in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                # Compute mean for numeric columns
                mean_value = dataframe[col].mean()
                outputDataFrame[col] = [mean_value]
                # print(mean_value)
            else:
                # Get the last non-null value for non-numeric columns
                last_value = dataframe[col].dropna().iat[-1]
                outputDataFrame[col] = [last_value]

        # Resulting DataFrame should have one row and each column representing the computed values
        return outputDataFrame

        # print(listOfEvaluators)

        # for evaluator in listOfEvaluators:

        #     response = ""
        #     self.prompt = None
        #     self.prompt = self.getLLMPrompt(evaluator)

        #     if(self.prompt != None):

        #         if(self.isInBetaMode == True):

        #             response = self.getEvalutorResponse(evaluator)

        #         else:

        #             response = self.llm.getResponse(prompt=self.prompt)['choices'][0]['message']['content']

        #     else:
                
        #         raise Exception("Critical Failure: No Prompt Found For: " + str(evaluator))

        #     if len(response) != 0:

        #         evaluationDataList = self.getEvalutorData(evaluator)
        #         evaluationDataList.insert(0, response)
        #         evaluationModule = EvaluationMethodFactory.create_evaluation(evaluator, evaluationDataList)
        #         listOfEvaluationModules.append(evaluationModule)

        #     else: 

        #         raise Exception("Critical Failure: No Response From LLM Found")

        # # with multiprocessing.Pool(processes=len(listOfEvaluationModules)) as pool:
        # #     # Map the evaluate_instance function to each class instance
        # #     results = pool.map(self.callEvaluator, listOfEvaluationModules)
        # #     print(results)

        # results = []

        # for process in listOfEvaluationModules:
        #     results.append(process.PerformEvaluation())

        # df = pd.DataFrame()
        # for result in results:

        #     if isinstance(result, dict):
                
        #         keys = result.keys()
        #         for key in keys:

        #             df[key] = [str(result[key])]
        #             print(key)
        #             print(str(result[key]))

        #     else:

        #         if isinstance(result[1], dict):

        #             df[result[0]] = [str(result[1])]
        #             print(result[0])
        #             print(str(result[1]))

        #         else:

        #             df[result[0]] = [result[1]]
        #             print(result[0])
        #             print(result[1])

        # print(df.head())

        # df['CurrentDate'] = date.today()
        # df.to_csv("results.csv", index=False)
        # #callBackend(listOfEvaluators, results)
    # def custom_mean(self, series):
    #     if pd.api.types.is_numeric_dtype(series):  # If numeric, compute the mean
    #         return series.mean()
        
    #     # Convert all values to tuples (hashable) for comparison if they are lists or unhashable types
    #     unique_values = series.dropna().apply(lambda x: tuple(x) if isinstance(x, list) else x).nunique()
        
    #     return unique_values[0] if len(unique_values) == 1 else None  # Return value if unique, else None




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
        Input: N/A
        Output: Prompt DataFrame
    """
    def getPrompts(self):
                
        with open(os.path.join("prompts", "prompts.txt"), 'r') as promptFile:
            lines = promptFile.readlines()[1:]

        promptDF = pd.DataFrame([line.strip().split(",") for line in lines], columns=["prompt", "data", "type"])

        return promptDF
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