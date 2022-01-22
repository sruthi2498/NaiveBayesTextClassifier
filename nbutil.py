import json 
import math 
def dumpVocab(vocab):
    with open("vocab.txt","w") as f:
        f.write("\n".join(vocab))

def getVocab():
    with open('vocab.txt') as f:
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

def getTokenClassConditionalLogProb(model,tokens, class_type):
    res=1
    for t in tokens:
        if t in model:
            res+=math.log(model[t][class_type],2)
    return res 

def getPredictions(test_data,model, predKnown =  False):
    pos_prior,neg_prior,tru_prior,dec_prior = getPriors(model)
    class1_pred_count = 0
    class2_pred_count = 0
    wrong_preds1=[]
    wrong_preds2 = []
    result = []
    for i in range(len(test_data)):
        row = test_data[i]
        '''
        pos_posterior_prob = getTokenClassConditionalProb(model,row["tokens"],"pos_prob")*pos_prior
        neg_posterior_prob = getTokenClassConditionalProb(model,row["tokens"],"neg_prob")*neg_prior

        tru_posterior_prob = getTokenClassConditionalProb(model,row["tokens"],"tru_prob")*tru_prior
        dec_posterior_prob = getTokenClassConditionalProb(model,row["tokens"],"dec_prob")*dec_prior'''

        pos_posterior_prob = getTokenClassConditionalLogProb(model,row["tokens"],"pos_prob")+math.log(pos_prior,2)
        neg_posterior_prob = getTokenClassConditionalLogProb(model,row["tokens"],"neg_prob")+math.log(neg_prior,2)
        tru_posterior_prob = getTokenClassConditionalLogProb(model,row["tokens"],"tru_prob")+math.log(tru_prior,2)
        dec_posterior_prob = getTokenClassConditionalLogProb(model,row["tokens"],"dec_prob")+math.log(dec_prior,2)
        
        class1_pred_label = 1 if pos_posterior_prob>=neg_posterior_prob else 0
        if predKnown :
            if class1_pred_label==row["class1"]:
                class1_pred_count+=1
            else:
                wrong_preds1.append({"text":row["text"], "label":row["class1"], "pred":class1_pred_label, "pos_posterior_prob":pos_posterior_prob,"neg_posterior_prob":neg_posterior_prob})
        class1_pred_label = "positive" if class1_pred_label==1 else "negative"

        class2_pred_label = 1 if tru_posterior_prob>=dec_posterior_prob else 0
        if predKnown:
            if class2_pred_label==row["class2"]:
                class2_pred_count+=1
            else:
                wrong_preds2.append({"text":row["text"], "label":row["class2"], "pred":class2_pred_label})
        class2_pred_label = "truthful" if class2_pred_label==1 else "deceptive"

        result.append(class2_pred_label+" "+class1_pred_label+" "+row["filename"])

    if predKnown:
        print(class1_pred_count,len(test_data),class1_pred_count/len(test_data))
        print(class2_pred_count,len(test_data),class2_pred_count/len(test_data))
    return wrong_preds1,wrong_preds2,result

def dumpResult(result, filename):
    with open(filename,"w") as f:
        f.write("\n".join(result))

