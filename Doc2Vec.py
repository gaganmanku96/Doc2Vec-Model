import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import TaggedDocument, LabeledSentence
from gensim.models import Doc2Vec
from gensim import utils


# Iterator for Tagging Documents
class LabeledLineSentence():
    def __init__(self,fileName):
        self.fileName = fileName
        
    def __iter__(self):
        df = pd.read_csv(self.fileName)
        text = df['text'].values
        for idx, doc in tqdm(enumerate(text)):
            doc = self.preprocess(doc)
            yield TaggedDocument(words=doc.split(),tags=[idx])
            
    def preprocess(self, doc):
        doc = re.sub('[^a-z]',' ',doc.lower())
        return doc


iterator = LabeledLineSentence('food_wine_raw.csv')
model = Doc2Vec(iterator,min_count=1, window=3, vector_size=50, sample=1e-4, negative=5, workers=3)

model.train(iterator,total_examples=model.corpus_count,epochs=30)

print(model.wv.most_similar('wine'))

rating = []
df = pd.read_csv('food_wine_raw.csv')

rules = {1:0,2:0,3:0,4:1,5:1}
df['rating'] = df['rating'].replace(rules)
rating = df['rating'].values
del df

train_arrays = np.ones((2000, 50))
train_labels = np.ones(2000,dtype='int')

test_arrays = np.zeros((444, 50))
test_labels = np.zeros(444,dtype='int')

for i in range(2000):
    train_arrays[i] = model[i]
    train_labels[i] = rating[i]   
    
for i in range(444):
    test_arrays[i] = model[2000+i]
    test_labels[i] = rating[2000+i]   

classifier = LogisticRegression(solver='lbfgs',C=0.5)
classifier.fit(train_arrays,train_labels)
pred = classifier.predict(test_arrays)   

cm = confusion_matrix(test_labels,pred)
accuracy = accuracy_score(test_labels,pred)

print(accuracy)


