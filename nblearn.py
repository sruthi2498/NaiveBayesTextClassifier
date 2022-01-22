import sys 
import nbpreprocess
import nbutil
import numpy as np

path_to_input = sys.argv[1]
folds = 4

data = nbpreprocess.extractData(path_to_input, train = True)
print("total data :",len(data))

def extractCompleteVocab(data):
    all_tokens = [d["tokens"] for d in data]
    all_tokens = [item for l in all_tokens for item in l ]
    all_tokens = sorted(list(set(all_tokens)))
    return all_tokens

vocab = extractCompleteVocab(data)

print("vocab : ",len(vocab))
nbutil.dumpVocab(vocab)


dev_data = [d for d in data if d["foldNum"]==1]
train_data = [d for d in data if d["foldNum"]!=1]


def getDataArr(data):
    data_arr = np.zeros((len(data),len(vocab)+2))
    print(data_arr.shape)
    for i,row in enumerate(data):
        for token in row["tokens"]:
            token_index = vocab.index(token)
            data_arr[i][token_index] = 1
        data_arr[i][-2] = row["class1"]
        data_arr[i][-1] = row["class2"]
    # print(class1_data)
    return data_arr


def calculatePrior(train_data_arr, model):
    total_pos_count = (train_data_arr[:,-2] == 1).sum()
    total_neg_count = (train_data_arr[:,-2] == 0).sum()
    print("pos count:",total_pos_count,"neg count:",total_neg_count)

    model["pos_prob"] = total_pos_count/(total_pos_count+total_neg_count)
    model["neg_prob"] = total_neg_count/(total_pos_count+total_neg_count)

    total_tru_count = (train_data_arr[:,-1] == 1).sum()
    total_dec_count = (train_data_arr[:,-1] == 0).sum()
    print("truth count:",total_tru_count,"decep count:",total_dec_count)

    model["tru_prob"] = total_tru_count/(total_tru_count+total_dec_count)
    model["dec_prob"] = total_dec_count/(total_tru_count+total_dec_count)
    return model


def calculateVocabForEachClass(train_data_arr):
    all_pos_words = set()
    all_neg_words = set()
    all_tru_words = set()
    all_dec_words = set()

    for i in range(len(train_data_arr)):
        if train_data_arr[i][-2] == 1:
            for j in range(len(vocab)):
                if train_data_arr[i][j] and vocab[j] not in all_pos_words:
                    all_pos_words.add(vocab[j])
        else:
            for j in range(len(vocab)):
                if train_data_arr[i][j] and vocab[j] not in all_neg_words:
                    all_neg_words.add(vocab[j])

        if train_data_arr[i][-1] == 1:
            for j in range(len(vocab)):
                if train_data_arr[i][j] and vocab[j] not in all_tru_words:
                    all_tru_words.add(vocab[j])
        else:
            for j in range(len(vocab)):
                if train_data_arr[i][j] and vocab[j] not in all_dec_words:
                    all_dec_words.add(vocab[j])
        


    num_pos_words  = len(all_pos_words)
    num_neg_words  = len(all_neg_words)
    num_tru_words  = len(all_tru_words)
    num_dec_words  = len(all_dec_words)

    print("pos words count:",num_pos_words,"neg words count:",num_neg_words)
    print("tru words count:",num_tru_words,"dec words count:",num_dec_words)

    return num_pos_words,num_neg_words,num_tru_words,num_dec_words



def calculateWordClassConditionalProb(train_data_arr, model,num_pos_words,num_neg_words,num_tru_words,num_dec_words):
    for j,word in enumerate(vocab):
        pos_count = 0
        neg_count = 0
        tru_count = 0
        dec_count = 0
        for i in range(len(train_data_arr)):
            if train_data_arr[i][j]==1:
                if train_data_arr[i][-2]==1:
                    pos_count+=1
                else:
                    neg_count+=1

                if train_data_arr[i][-1]==1:
                    tru_count+=1
                else:
                    dec_count+=1
        model[word] = {"pos_prob":0,"neg_prob":0,"tru_prob":0,"dec_prob":0}

        model[word]["pos_prob"] =pos_count/num_pos_words if pos_count else 1/num_pos_words
        model[word]["neg_prob"] =neg_count/num_neg_words if neg_count else 1/num_neg_words
        model[word]["tru_prob"] =tru_count/num_tru_words if tru_count else 1/num_tru_words
        model[word]["dec_prob"] =dec_count/num_dec_words if dec_count else 1/num_dec_words
    return model 


train_data_arr = getDataArr(train_data)
model = {}
model = calculatePrior(train_data_arr,model)
num_pos_words,num_neg_words,num_tru_words,num_dec_words = calculateVocabForEachClass(train_data_arr)
model = calculateWordClassConditionalProb(train_data_arr,model,num_pos_words,num_neg_words,num_tru_words,num_dec_words)
nbutil.dumpModel(model)


print("Test on dev data")

test_data = dev_data
model = nbutil.getModel()
result = nbutil.getPredictions(test_data,model,predKnown=True)


train_data_arr = getDataArr(data)
model = {}
model = calculatePrior(train_data_arr,model)
num_pos_words,num_neg_words,num_tru_words,num_dec_words = calculateVocabForEachClass(train_data_arr)
model = calculateWordClassConditionalProb(train_data_arr,model,num_pos_words,num_neg_words,num_tru_words,num_dec_words)
nbutil.dumpModel(model)


