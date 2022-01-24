import os 
import string
from difflib import get_close_matches,SequenceMatcher
import nbutil
import json
import time

punctuations = set(list(string.punctuation))

with open("stopwords.txt") as f:
    stopwords = set([w.replace("\n","") for w in f.readlines()])


def tokenizeText(text):
    text = text.lower().strip().rstrip().lstrip().replace("\t","")
    for punct in punctuations:
        text = text.replace(punct,"")
    words = list(set(text.split(" ")))
    tokens = [w for w in words if w not in stopwords and w.isalpha() and len(w)>1]
    return tokens


def extractData(path_to_input, train=True):
    data_array = []
    class_folders = os.listdir(path_to_input)
    for main_dir in class_folders:
        main_dir = path_to_input+"/"+main_dir+"/"
        if os.path.isdir(main_dir):
            children = os.listdir(main_dir)

            class1_label = 0 if "negative" in main_dir else 1

            for child in children:
                path = main_dir + child +"/"

                class2_label = 1 if "truthful" in path else 0

                if os.path.isdir(path) :
                    fold_dirs = os.listdir(path)
                    foldNum = 1
                    for foldDir in fold_dirs:
                        if os.path.isdir(path+foldDir):
                            files= os.listdir(path+foldDir)
                            for f in files:
                                with open(path+foldDir+"/"+f) as fp:
                                    lines = fp.readlines()
                                    text = "".join(lines).strip().rstrip().lstrip()
                                    tokens = tokenizeText(text)
                                    # data_array.append([text,tokens,foldNum,class1_label,class2_label,fold_dir+f])
                                    row = {
                                        "text":text,
                                        "tokens":tokens,
                                        "foldNum":foldNum,
                                        "filename":path+foldDir+"/"+f,
                                        }
                                    if train:
                                        row["class1"] = class1_label
                                        row["class2"] = class2_label
                                    data_array.append(row)
                        foldNum+=1
                    
    return data_array

def exploreVocab():
    vocab = nbutil.getVocab()
    for word in vocab[20:40]:
        print(word, get_close_matches(word,vocab[20:40]))
        for word2 in vocab[20:40]:
            if word2!=word:
                print("score for: " + word + " vs. " + word2 + " = " + str(SequenceMatcher(None, word, word2).ratio()))

def countVocabOccurrences(vocab,data):
    vocab_count={}
    for w in vocab:
        vocab_count[w]=0
        for i in range(len(data)):
            for t in data[i]["tokens"]:
                if w==t:
                    vocab_count[w]+=1
    return sorted(vocab_count.items(), key=lambda x : x[1],reverse=True)

def getVocabCloseMatches():
    vocab = nbutil.getVocab()
    marked = [False]*len(vocab)
    closest_match = {}
    t1= time.time()
    for i,word in enumerate(vocab):
        if not marked[i]:
            marked[i] = True
            closest_match[word] = []
            word_matches = get_close_matches(word,vocab, n=10)
            for word2 in word_matches:
                if word2!=word:
                    score = SequenceMatcher(None, word, word2).ratio()
                    if score>0.89:
                        marked[vocab.index(word2)]= True
                        closest_match[word].append(word2)
            if not closest_match[word]:
                closest_match.pop(word)


    # print(closest_match)

    closest_word_map = {}
    for k,v in closest_match.items():
        for w in v:
            closest_word_map[w] = k

    with open("closest_word_map.txt","w") as f:
        json.dump(closest_word_map,f)
    print(time.time()-t1)