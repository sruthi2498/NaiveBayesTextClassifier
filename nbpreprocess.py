from json.tool import main
import os 
import string

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

    


def extractTestingData(path_to_input, folds):
    data_array = []
    class_folders = os.listdir(path_to_input)
    for main_dir in class_folders:
        main_dir = path_to_input+"/"+main_dir+"/"
        # print(main_dir, os.path.isdir(main_dir))
        if os.path.isdir(main_dir) :
            children = os.listdir(main_dir)
            for child in children:
                path = main_dir + child
                if os.path.isdir(path) :
                    # print(path)
                    for foldNum in range(1,folds+1):
                        fold_dir = path+"/fold"+str(foldNum)+"/"
                        files = os.listdir(fold_dir)
                        for f in files:
                            with open(fold_dir+f) as fp:
                                lines = fp.readlines()
                                text = "".join(lines).strip().rstrip().lstrip()
                                tokens = tokenizeText(text)
                                row = {
                                    "text":text,
                                    "tokens":tokens,
                                    "foldNum":foldNum,
                                    "filename":fold_dir+f,
                                    }

                                data_array.append(row)
    return data_array

    