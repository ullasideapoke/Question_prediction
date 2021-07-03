# import libaraies

from flask import Flask, render_template, request, redirect, url_for
from gingerit.gingerit import GingerIt
from pycorenlp import StanfordCoreNLP
from Co_referencer import coreference_pipeline

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
import spacy
import ast
import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
parser = GingerIt()

app = Flask(__name__)

# model1: StanfordCoreNLP

class isQuestionBasic():
    
    # Init Constructor
    # Initialize StanfordCore NLP local instance on port 9000
    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        
    # Input: Sentence to be predicted
    # Processing: 1. Uses Stanfors NLP's 'annotate' method to create Parse Tree
    # 2. Checks for occurence of 'SQ' or 'SBARQ' in the parse tree
    # Return: 1 - If sentence is question | 0 - If sentence is not a question


    def isQuestion(self, sentence):
        try:
            if '?' in sentence:
                return 1
            output = self.nlp.annotate(sentence, properties={
                'annotators': 'parse',
                'outputFormat': 'json',
                'timeout': 1000,
            })

            if ('SQ' or 'SBARQ') in output['sentences'][0]["parse"]:
                return 1    
            else:
                return 0
        except:
            return 0


isQuestionBasic_obj = isQuestionBasic()



# Model 2: BertTokenizer -  for those which stanford was unable to predict as questions


# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')


# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')



class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
        super(BERT_Arch, self).__init__()

        self.bert = bert 

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask,return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

#load weights of best model
path = './resources/new_saved_weights.pt'
model.load_state_dict(torch.load(path))
max_seq_len = 25




#Functions defining models 

def stanford_question_identification(input_text):
    result = isQuestionBasic_obj.isQuestion(input_text)
    # print("stanford_results",result)
    return result

def bert_question_identification(input_text):

    tokens_result = tokenizer.batch_encode_plus(
    [input_text],
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False)
    
    result_seq = torch.tensor(tokens_result['input_ids'])
    result_mask = torch.tensor(tokens_result['attention_mask'])
 
    with torch.no_grad():
        preds = model(result_seq, result_mask)
        preds = preds.detach().numpy()


    # model's performance
    preds = np.argmax(preds, axis = 1)
    # print("Bert_results:",preds)


    if preds[0] == 0:
        preds = "Not a question"
    else:
        preds = "it's a question"
    
    return preds

# Main Functions

def question_identification(input_sentences):

    predicted_question_list = []
    try:
        
        input_sentences = parser.parse(input_sentences) #spell check
        input_sentences = dict((k, input_sentences[k]) for k in ['result'] if k in input_sentences)
        input_sentences= " ".join(input_sentences.values())
        
        print("input_text before coreference", input_sentences)
        input_sentences = coreference_pipeline(input_sentences)
        print("input_text after coreference", input_sentences)

        doc = nlp(input_sentences)
        
        for sent in doc.sents:
            
            input_text = sent.text           

            stanford_result = stanford_question_identification(input_text)

            if stanford_result == 1:
                # print("result obtained using stanford method")
                final_result = {input_text: "it's a question"}
                # print("stan", final_result)

            else:
                # print("result obtained using bert method")
                bert_result = bert_question_identification(input_text)
                final_result = {input_text: bert_result}
                # print("bert", final_result)
            
            if list(final_result.values())[0] == "it's a question":
                predicted_question_list.append(final_result)


        return predicted_question_list
    
    except Exception as exp:
        return predicted_question_list
        print("question_identification_main")
        print(exp)



@app.route('/question_identification', methods=['POST'])

# Pipeline function for implementation

def question_identification_pipeline():
    result_list=[]
    try:
        if request.method == 'POST':
            req_json = ast.literal_eval(request.get_data(as_text=True))
            input_sentences = str(req_json["input_sentences"])
            result_list = question_identification(input_sentences)

    except Exception as exp:
        print("question_identification_pipeline")
        print(exp)

    return {"output":result_list}


if __name__ == '__main__':

        app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)
    
        # input_text = "i want to be able to confirm whether the cost and manufacturing speed are the current values or the values when a certain quantity is made in the future"
        # input_text ="We have gone throught the information provided about startups technical and technological information. But, the information was limited and also common information available on net. we would like to get more deep information regarding the Startups and few more startups can be included."
        # output= question_identification(input_text)
        # print(output)

