# Importing essential libraries
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize

# Loading the dataset
df = pd.read_csv(r"C:\Users\Admin\Downloads\emails", sep=",")

corpus = []
ps = PorterStemmer()
for i in range(0, df.shape[0]):
    content = re.sub(pattern = '[^a-zA-Z]', repl=' ', string = new_df.content[i])
    content = content.lower()
    words = content.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    words = [ps.stem(word) for word in words]
    title = ' '.join(words)
    corpus.append(title)

new_corp = pd.read_csv("Project_NLP_EDA_clean_data.csv")
new_corp.head()
new_corp.isna().sum()
new_corp.dropna(inplace=True)
new_corp.reset_index(inplace=True)

corpus = []
for i in range(0, new_corp.shape[0]):
    content = new_corp.content[i]
    words = content.split()
    title = ' '.join(words)
    corpus.append(title)
    
#Extract Feature With CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray() # Fit the Data
   
pickle.dump(cv, open('tranform.pkl', 'wb'))

class_values = pd.get_dummies(new_corp['Class'])
class_values = class_values.drop(columns="Non Abusive")

y = class_values.values.ravel()

  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Logistic regression Classifier
from sklearn.linear_model import LogisticRegression

clf =  LogisticRegression(max_iter=500, random_state=0)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
