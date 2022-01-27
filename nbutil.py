import json 
import math 
import numpy as np
import re

def dumpVocab(vocab):
    filename = "vocab.txt"
    with open(filename,"w") as f:
        f.write("\n".join(vocab))

def getVocab():
    filename = "vocab.txt"
    
    with open(filename) as f:
        vocab = [x.replace("\n","") for x in f.readlines()]
        return vocab

def dumpModel(model):
    with open('nbmodel.txt', 'w') as f:
        json.dump(model, f)

def getModel():
    with open('nbmodel.txt') as f:
        model = json.load( f)
        return model

def getPriors(model):
    pos_prior = model["pos_prob"]
    neg_prior = model["neg_prob"]
    tru_prior = model["tru_prob"]
    dec_prior = model["dec_prob"]
    return pos_prior,neg_prior,tru_prior,dec_prior


def getTokenClassConditionalProb(model,tokens, class_type):
    res=1
    for t in tokens:
        if t in model:
            res*=model[t][class_type]
    return res 

def getTokenClassConditionalLogProb(model,vocab,tokens, class_type):
    res=1
    for t in tokens:
        t = getBaseWord(vocab,t)
        if t in model:
            res+=math.log(model[t][class_type],2)
    return res 

def getPredictions(test_data,model,vocab, predKnown =  False):
    pos_prior,neg_prior,tru_prior,dec_prior = getPriors(model)
    tp1,fp1,tn1,fn1 = 0,0,0,0
    tp2,fp2,tn2,fn2 = 0,0,0,0
    result = []
    for i in range(len(test_data)):
        row = test_data[i]
        '''
        pos_posterior_prob = getTokenClassConditionalProb(model,row["tokens"],"pos_prob")*pos_prior
        neg_posterior_prob = getTokenClassConditionalProb(model,row["tokens"],"neg_prob")*neg_prior

        tru_posterior_prob = getTokenClassConditionalProb(model,row["tokens"],"tru_prob")*tru_prior
        dec_posterior_prob = getTokenClassConditionalProb(model,row["tokens"],"dec_prob")*dec_prior'''

        pos_posterior_prob = getTokenClassConditionalLogProb(model,vocab,row["tokens"],"pos_prob")+math.log(pos_prior,2)
        neg_posterior_prob = getTokenClassConditionalLogProb(model,vocab,row["tokens"],"neg_prob")+math.log(neg_prior,2)
        tru_posterior_prob = getTokenClassConditionalLogProb(model,vocab,row["tokens"],"tru_prob")+math.log(tru_prior,2)
        dec_posterior_prob = getTokenClassConditionalLogProb(model,vocab,row["tokens"],"dec_prob")+math.log(dec_prior,2)
        
        class1_pred_label = 1 if pos_posterior_prob>=neg_posterior_prob else 0
        if predKnown:
            if row["class1"]==1:
                if class1_pred_label==1:
                    tp1+=1
                else:
                    fn1+=1
            else:
                if class1_pred_label==0:
                    tn1+=1
                else:
                    fp1+=1

        class1_pred_label = "positive" if class1_pred_label==1 else "negative"

        class2_pred_label = 1 if tru_posterior_prob>=dec_posterior_prob else 0
        if predKnown:
            if row["class2"]==1:
                if class2_pred_label==1:
                    tp2+=1
                else:
                    fn2+=1
            else:
                if class2_pred_label==0:
                    tn2+=1
                else:
                    fp2+=1

        class2_pred_label = "truthful" if class2_pred_label==1 else "deceptive"

        result.append(class2_pred_label+" "+class1_pred_label+" "+row["filename"])

    if predKnown:
        precision1 = tp1/(tp1+fp1)
        recall1 = tp1/(tp1+fn1)
        fscore1 = (2*precision1*recall1)/(precision1+recall1)
        acc1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)

        precision2 = tp2/(tp2+fp2)
        recall2 = tp2/(tp2+fn2)
        fscore2 = (2*precision2*recall2)/(precision2+recall2)
        acc2 = (tp2+tn2)/(tp2+tn2+fp2+fn2)
        
        print(precision1,recall1,fscore1,acc1)
        print(precision2,recall2,fscore2,acc2)
        print("avg f1",(fscore1+fscore2)/2)
    return result

def dumpResult(result, filename):
    with open(filename,"w") as f:
        f.write("\n".join(result))

def countVocabOccurrences(vocab,data):
    vocab_count={}
    for w in vocab:
        vocab_count[w]=0
        for i in range(len(data)):
            for t in data[i]["tokens"]:
                if w==t:
                    vocab_count[w]+=1
    return sorted(vocab_count.items(), key=lambda x : x[1],reverse=True)

def getBaseWord(vocab,word):
    if len(word)<=1:
        return word
    if (word[-1]=="s" or word[-1]=="y" or word[-1]=="d") and word[:-1] in vocab:
        return word[:-1]
    if (word[-2:]=="ly" or word[-2:]=="ed") and word[:-2] in vocab:
        return word[:-2]
    if len(word)>3 and (word[-3:]=="ing" or word[-3:]=="ful") and word[:-3] in vocab :
        return word[:-3]
    if len(word)>3 and (word[-3:]=="ing" or word[-3:]=="ion") and word[:-3]+"e" in vocab :
        return word[:-3]+"e"
    if len(word)>3 and (word[-3:]=="ing" or word[-3:]=="ion") and word[:-3]+"ed" in vocab :
        return word[:-3]+"ed"
    if len(word)>4 and word[-4:]=="ions" and word[:-4]+"e" in vocab :
        return word[:-4]+"e"
    if len(word)>4 and word[:4]=="well" and word[4:] in vocab :
        return word[4:]
    if word[-1]=="y" and word[:-1]+"e" in vocab:
        return word[:-1]+"e"
    if len(word)>3 and word[-3:]=="ies" and word[:-3]+"y" in vocab :
        return word[:-3]+"y"
    word= re.sub(r'\bhapp(ily|y|ier|iness)',"happy",word)
    word= re.sub(r'\bresponsib(ility|le)',"responsible",word)

    return word
    