import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from abc import ABC, abstractmethod
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import torch
import jiwer
# from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer
from nltk.corpus import stopwords
from nltk import download
from nltk.tokenize import word_tokenize
import nltk
import lexical_diversity as ld
from gensim.models import Word2Vec, KeyedVectors
from rouge_score import rouge_scorer

# from evaluatorDependencies.FactcheckGPT.src.pipeline import check_document
# import fasttext
# import factscore
# import evaluatorDependencies.chrF as characterLevelScore

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
        # elif dataType == "perp":
        #     return perpEvaluator(data)
        elif dataType == "meteor":
            return meteorEvaluator(data)
        elif dataType == "rouge":
            return rougeEvaluator(data)
        elif dataType == "wer":
            return werEvalutor(data)
        elif dataType == "editDistance":
            return editDistanceEvalutor(data)
        elif dataType == "bert":
            return bertEvaluator(data)
        # elif dataType == "ttrEvaluator":
        #     return ttrEvaluator(data)
        # elif dataType == "wordMover":
        #     return wordMoverEvaluator(data)
        else:
            print("Only Testing the most basic evaluators currently not: " + str(dataType))
            # raise ValueError("Unsupported data type for evaluation")

class Evaluator(ABC):
    @abstractmethod
    def PerformEvaluation(self):
        pass

def getEvaluators():
    return [
        # "perp",
        "bleu",
        "rouge",
        "meteor",
        # "chrf",
        "wer",
        "editDistance",
        "bert",
        # "wordMover",
        # "ttr",
        # "mtld",
        # "hdd",
        # "factScorer"

    ]

"""
    Data will be in the formate: [LLM Response, Data String]
"""
class bleuEvaluator(Evaluator):

    def __init__(self, data):
        self.data = data
    
    def PerformEvaluation(self):
        
        score = 0

        score = sentence_bleu([self.data[0].strip().split()], self.data[1].strip().split())

        return ("bleu", score)

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
    Data will be in the format: [llmResponse, DataString]
"""
class rougeEvaluator(Evaluator):

    def __init__(self, data):

        self.data = data
    
    def PerformEvaluation(self):

        rouge_score = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        return rouge_score.score(self.data[0], self.data[1])

"""
    Data will be in the format: [LLM Output, Data String]
"""
class meteorEvaluator(Evaluator):

    def __init__(self, data):

        self.data = data
    
    def PerformEvaluation(self):

        nltk.download('punkt_tab')
        nltk.download('wordnet')

        return ("meteor", meteor_score([word_tokenize(self.data[0])], word_tokenize(self.data[1])))

"""
    Data will be in the format: [Base truth string, LLM Output string]
"""
class chrFEvaluator(Evaluator):

    def __init__(self, data):

        self.data = data
    
    def PerformEvaluation(self):

        print("TODO: Fixing")
        # return characterLevelScore.chrF(self.data[1], self.data[2])

"""
    Data will be in the format: [LLM Output String, Base truth string]
"""
class werEvalutor(Evaluator):

    def __init__(self, data):

        self.data = data
    
    def PerformEvaluation(self):

        transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ])
        
        return ("wer", jiwer.wer(self.data[1], self.data[0], truth_transform=transforms, hypothesis_transform=transforms))

"""
    Data will be in the following format [LLM Output String, Truth String]
"""
class editDistanceEvalutor(Evaluator):

    def __init__(self, data):

        self.data = data
    
    def PerformEvaluation(self):

        s1 = self.data[0]
        s2 = self.data[1]

        m, n = len(s1), len(s2)

        # Create a table to store results of subproblems
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill the known entries in dp[][]
        # If one string is empty, then answer 
        # is length of the other string
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill the rest of dp[][]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

        return ("editDistance", dp[m][n])
    
"""
    Data will be in the format: [Base truth string, LLM Output string]
    Output: A dictionary with each value and its corresponding evaluation metric
"""
class bertEvaluator(Evaluator):

    def __init__(self, data):

        self.data = data
    
    def PerformEvaluation(self):

        P, R, F1 = BERTScorer(model_type='bert-base-uncased').score([self.data[1]], [self.data[0]])
        return ("bert", {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1":F1.mean().item()})
    
"""
    Data will be in the format: [LLM Output string, Base truth string,]
"""
class wordMoverEvaluator(Evaluator):

    def __init__(self, data):

        self.data = [data[0].lower().split(" "),data[1].lower().split(" ")]
    
    def PerformEvaluation(self):

        nltk.download("stopwords")
        stop_words = stopwords.words("english")
        cleanedBaseSentence = [w for w in self.data[1] if w not in stop_words]
        cleanedHypothesisSentence = [w for w in self.data[0] if w not in stop_words]
        
        fasttext.util.download_model('en', if_exists='ignore')
        fasttext.load_model('cc.en.300.bin')

        model = KeyedVectors.load_word2vec_format('cc.en.100.bin', binary=True)

        return ("wordMover", model.wmdistance(cleanedBaseSentence, cleanedHypothesisSentence))
    
# """
#     The fact scorer method is unique as it is brand new and uses a model trained on wikipedia for accuracy requires this model to be self contained
#     Data will be in the formate [topic, LLM output string, llmAPIKey]
# """
class factScorerEvaluator(Evaluator):

    def __init__(self, data):

        self.data = list([data[0]]), list([data[1]], data[2])
    
    def PerformEvaluation(self):

        print("TODO: Fixing")
        # fs = FS.FactScorer(openai_key = self.data[2])
        # return fs.get_score(self.data[0], self.data[1], gamma=1)["score"]
"""
    Data will be in the format: [List Of Strings]
"""
class ttrEvaluator(Evaluator):

    def init(self, data):

        self.data = data

    def PerformEvaluation(self):
        ld_object = ld.LexicalDiversity(self.data)

        ttr = ld_object.ttr

        return ttr

"""
    Data will be in the format: [List Of Strings]
"""
class mtldEvaluator(Evaluator):

    def init(self, data):

        self.data = data

    def PerformEvaluation(self):
        ld_object = ld.LexicalDiversity(self.data)

        mtld = ld_object.mtld()

        return mtld

"""
    Data will be in the format: [List Of Strings]
"""
class hddEvaluator(Evaluator):

    def init(self, data):

        self.data = data

    def PerformEvaluation(self):
        ld_object = ld.LexicalDiversity(self.data)

        hdd = ld_object.hdd()

        return hdd