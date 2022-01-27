import os 
import string
import re
import nbutil

punctuations = set(list(string.punctuation))

def replaceStopwords(word):
    word = re.sub(r'\bno(r|t|w)*\b',"",word)
    word = re.sub(r'\bon(ce|ly)*\b',"",word)
    word = re.sub(r'\byou[\']*(re|ll|r|ve|self|selves|d)\b', '', word)
    word = re.sub(r'\bagain(st)*\b', '', word)
    word = re.sub(r'\b(would|should|could|did|need|are|were|was|must|have|has|had|might|must|does|do|shan|is|won)[\'|\"]*n*[\'|\"]*t*(ve)*\b', '', word)
    word = re.sub(r'\ba[m|n][d|y]\b',"",word)
    word = re.sub(r'\bwh(ere|at|en|ich|y|ile|hom)\b',"",word)
    word = re.sub(r'\b(her|him|his|my)(self)*\b',"",word)
    word = re.sub(r'\b(her)(s|e)*\b',"",word)
    word = re.sub(r'\b(she)[\']*(s)*\b',"",word)
    word = re.sub(r'\b(it|our|them)[\']*(s|self|selves)*\b',"",word)
    word = re.sub(r'\b(that)[\']*(ll)*\b',"",word)
    word = re.sub(r'\b(the)(ir|re|se|s|y|n)*\b',"",word)
    word = re.sub(r'\b(th)(is|ose)\b',"",word)
    word = re.sub(r'\b(i)[\'|\"]*(d|ll|m|ve|f|m|n|t|nto)\b',"",word)
    word = re.sub(r'\b(to)o*\b',"",word)
    word = re.sub(r'\b([a-z])\1+\b',"",word)
    word = re.sub(r'\b(have|be|dur)(ing)\b',"",word)
    return word 




def tokenizeText(text):
    text = text.lower().strip().rstrip().lstrip().replace("\t","")
    for punct in punctuations:
        text = text.replace(punct,"")
    words = list(set(text.split(" ")))
    tokens = [ replaceStopwords(w) for w in words if  w.isalpha() and len(w)>1]
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


def retokenizeData(data,vocab):
    for i in range(len(data)):
        new_tokens = set()
        for t in data[i]["tokens"]:
            t = nbutil.getBaseWord(vocab,t)
            if t not in new_tokens:
                new_tokens.add(t)
        data[i]["tokens"] = list(new_tokens)
    return data