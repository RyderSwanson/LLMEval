from abc import ABC, abstractmethod
import evaluate
from nltk.translate.bleu_score import sentence_bleu
import torch

"""
    The Abstract EvaluationHandler Class which takes the prompt, response, and returns the evaluation metric class
"""
"""
    Input: 
    dataType, which is a string for the evaluator you wish to use
    data, the data for the evaluator specified has to be handled by evaluator handler first to ensure proper format
"""
class EvaluationMethodFactory:
    @staticmethod
    def create_evaluation(dataType, data):
        if dataType == "bleu":
            return bleuEvaluator(data)
        elif dataType == "perp":
            return perpEvaluator(data)
        elif dataType == "rouge":
            return rougeEvaluator(data)
        else:
            raise ValueError("Unsupported data type for evaluation")

class Evaluator(ABC):
    @abstractmethod
    def PerformEvaluation(self):
        pass


"""
    Data will be in the formate: [List Of Strings, List of strings with proper grammer]
"""
class bleuEvaluator(Evaluator):

    def __init__(self, data):
        self.data = data
    
    def PerformEvaluation(self):
        
        score = 0

        for i in range(len(self.data[0])):
            
            score += sentence_bleu([self.data[0][i].strip().split()], self.data[1][i].strip().split())

        score /= len(self.data[0])

        return score


"""
    Note: we have to have an unmasked llm for this perfomance metric to do anything

    format of data:
    [
    typeOfInput(discrete variable)0 if it is just the loss, 1 if it is a list of losses, 2 if it is a llm to loaded locally,
        
    if typeOfInput = 0 type num
        
        lossValue
        
    if typeOfInput = 1 type list
        [
            listOfLosses
        ]
    
    if typeOfInput = 2 type list
        [
        Prompt (string),
        Tokenizer,
        LLMmodel (model),
        ]
    ]
"""
class perpEvaluator(Evaluator):

    def __init__(self, data):

        self.data = data
    
    def PerformEvaluation(self):

        if(self.data[0][0] == 0):

            return torch.exp(self.data[0][1])
        
        elif(self.data[0][0] == 1):

            return torch.exp(sum(self.data[0][1]) / len(self.data[0][1]))

        elif(self.data[0][0] == 2):

            self.data[0][2].eval()

            inputs = self.data[0][1](self.data[0][0], return_tensors='pt')
    
            with torch.no_grad():

                outputs = self.data[0][2](**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

            perplexity = torch.exp(loss)
            return perplexity.item()

        else:

            raise ValueError("Unsupported type of input for perplexity evaluation")

"""
    Data will be in the format: [baseInformation, llmPrediction]
"""
class rougeEvaluator(Evaluator):

    def __init__(self, data):

        self.data = data
    
    def PerformEvaluation(self):

        rouge_score = evaluate.load("rouge")
        return rouge_score.compute(predictions=self.data[1], references=self.data[0], use_stemmer=True)

