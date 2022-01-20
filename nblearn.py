import sys 
import os 
import numpy as np
import string

path_to_input = sys.argv[1]
folds = 4
punctuations = set(list(string.punctuation))
with open("stopwords.txt") as f:
    stopwords = set([w.replace("\n","") for w in f.readlines()])



def extractData():
    data_array = []
    neg_folder = path_to_input+ "/negative_polarity/"
    pos_folder = path_to_input+ "/positive_polarity/"
    for main_dir in [neg_folder,pos_folder]:
        children = os.listdir(main_dir)
        class1_label = "negative" if "negative" in main_dir else "positive"
        for child in children:
            path = main_dir + child
            class2_label = "truthful" if "truthful" in path else "deceptive"
            if os.path.isdir(path) :
                for foldNum in range(1,folds+1):
                    fold_dir = path+"/fold"+str(foldNum)+"/"
                    files = os.listdir(fold_dir)
                    for f in files:
                        with open(fold_dir+f) as fp:
                            lines = fp.readlines()
                            text = "".join(lines).strip().rstrip().lstrip()
                            # data.append({"text":text,
                            #             "fold": foldNum,
                            #             "class1":class1_label,
                            #             "class2": class2_label,
                            #             "filename":fold_dir+f})
                            
                            tokens = tokenizeText(text)
                            data_array.append([text,tokens,foldNum,class1_label,class2_label,fold_dir+f])
    return data_array


def tokenizeText(text):
    text = text.lower().strip().rstrip().lstrip().replace("\t","")
    for punct in punctuations:
        text = text.replace(punct,"")
    words = list(set(text.split(" ")))
    tokens = [w for w in words if w not in stopwords and w.isalpha() and len(w)>0]
    return tokens


data = extractData()
print(len(data),len(data[0]))

# print(data[5])

def extractCompleteVocab(token_index,data):
    all_tokens = [d[token_index] for d in data]
    all_tokens = [item for l in all_tokens for item in l ]
    all_tokens = sorted(list(set(all_tokens)))
    return all_tokens

vocab = extractCompleteVocab(1,data)
print(len(vocab))

with open("vocab.txt","w") as f:
    f.write("\n".join(vocab))

