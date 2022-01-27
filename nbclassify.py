import re
import sys 
import nbpreprocess
import nbutil

path_to_input = sys.argv[1]
folds = 4

data = nbpreprocess.extractData(path_to_input, train=False)
print("total data :",len(data))

vocab = nbutil.getVocab()
print("vocab : ",len(vocab))

model = nbutil.getModel()
result = nbutil.getPredictions(data,model,vocab)
nbutil.dumpResult(result,"nboutput.txt")