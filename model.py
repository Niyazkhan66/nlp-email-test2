#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from wordcloud import WordCloud
import re # for handling string
import string # for handling mathematical operations
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from imblearn.combine import SMOTEENN 
from joblib import dump
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


data= pd.read_csv("F:/Project/NLP-Emails-abusive-nonabusive--master/P38-NLP-Emails/emails.csv")
# creating new dataframe using "content" and "class"
data.head()


# In[10]:


df= data.iloc[:,3:5]
df.head()


# In[11]:


df.info()


# In[12]:


duplicate= df[df.duplicated()]    # Checking and droping the duplicates
df2= df.drop_duplicates() 
df2.head()


# In[13]:


df2["Class"].value_counts()   # After droping duplicates - uniques values left


# In[14]:


df.isnull().sum()


# In[15]:


# Checking the unique values in the dataset
df.nunique()


# In[16]:


df["Class"].value_counts()


# In[17]:


sns.countplot(df["Class"], palette="hls")


# In[18]:


print(sum(df["Class"]=="Abusive")/sum(df["Class"]=="Non Abusive")*100, "percent of abusive emails")
print(100 - sum(df["Class"]=="Abusive")/sum(df["Class"]=="Non Abusive")*100, "percent of non-abusive emails")


# #### The count for "Abusive" class is very less compared to "Non Abusive" class, hence the data is imbalanced.

# In[19]:


# Seperating the "Abusive" and "Non Abusive" classes
abusive = df[df["Class"]=="Abusive"]
abusive.head()


# In[20]:


abusive.shape


# In[21]:


non_abusive = df[df["Class"]=="Non Abusive"]
non_abusive.head()


# In[22]:


non_abusive.shape


# In[23]:


# Balancing the data by removing the duplicate values from only "Non Abusive" class
non_abusive1 = non_abusive.drop_duplicates()


# In[24]:


non_abusive1.shape


# In[25]:


print("Abusive ", len(abusive),"\n""Non Abusive ", len(non_abusive1))


# In[26]:


print(len(abusive)/len(non_abusive1)*100, " percentage of abusive mails")
print(100- (len(abusive)/len(non_abusive1))*100, " percentage of non-abusive mails")


# #### The data is more balanced  compared to original data 

# In[27]:


# Concatinating both 'abusive' and 'non-abusive' data into single dataset
df1 = pd.concat([abusive,non_abusive1], axis=0, ignore_index=True)


# In[28]:


df1.head()


# In[29]:


df1.shape


# In[30]:


# text cleaning
df1['cleaned']=df1['content'].apply(lambda x: x.lower()) # remove lower cases
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('\w*\d\w*','', x)) # remove digits and words with digits
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)) # remove punctuation
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('\n'," ",x)) # remove extra spaces
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ',x)) # remove special characters
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub(r"http\\S+", " ",x)) # remove hyperlinks
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub(' +',' ',x)) # remove extra spaces
df1['cleaned']=df1['cleaned'].apply(lambda x: x.split('\n\n')[0])
df1['cleaned']=df1['cleaned'].apply(lambda x: x.split('\n')[0])
df1['cleaned'].head()


# In[31]:


# tokenise entire df
def identify_tokens(row):
    new = row['cleaned']
    tokens = nltk.word_tokenize(new)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df1['cleaned'] = df1.apply(identify_tokens, axis=1)
df1['cleaned']


# In[32]:


#lemmatization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

df1['lemma'] = df1['cleaned'].apply(lemmatize_text)
df1['lemma']


# In[33]:


# remove stopwords
#nltk.download('stopwords')
stop_words = []
with open("F:/Project/NLP-Emails-abusive-nonabusive--master/P38-NLP-Emails/stop.txt",encoding="utf8") as f:
    stop_words = f.read()
    
# getting list of stop words
stop_words = stop_words.split("\n")


# In[34]:


def remove_stops(row):
    my_list = row['lemma']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

df1['lemma_meaningful'] = df1.apply(remove_stops, axis=1)
df1['lemma_meaningful'].tail()


# In[35]:


# rejoin meaningful stem words in single string like a sentence
def rejoin_words(row):
    my_list = row['lemma_meaningful']
    joined_words = ( " ".join(my_list))
    return joined_words

df1['final'] = df1.apply(rejoin_words, axis=1)

spam= ' '.join(list(df1[df1['Class'] == "Abusive"]['final']))
ham= ' '.join(list(df1[df1['Class'] == "Non Abusive"]['final']))


# In[36]:


df1['final']


# In[37]:


df1.head()


# In[38]:


# # Preparing email texts into word count matrix format 
mail= df1.loc[:,['final','Class']]
mail['final'].replace('', np.nan, inplace=True)
mail.dropna(subset=['final'], inplace=True)
mail['label'] = mail['Class'].map({'Abusive': 0, 'Non Abusive': 1})
mail['label']


# In[39]:


df1.Class.value_counts()  


# In[40]:


mail.head()


# In[41]:


mail.isna().sum()


# In[42]:


from collections import Counter
word_list = ' '.join(mail["final"])
split_it = word_list.split() 


# In[43]:


Count = Counter(split_it) 
most_occur = Count.most_common(30) 


# In[44]:


# Most frequent words in the text 
most_occur


# In[45]:


# If more than 75% of emails have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(Count).most_common() if count > 18000 ]
add_stop_words


# In[46]:


resultwords  = [word for word in split_it if word not in add_stop_words]


# In[47]:


# Most occuring words after removing additional stop words
Count = Counter(resultwords) 

most_occur = Count.most_common(30) 
most_occur


# In[48]:


words_list = ' '.join(resultwords)


# In[49]:


# Worcloud of unique words in the dataset


# In[50]:


from wordcloud import WordCloud
unique_wordcloud = WordCloud(width=600, height=400).generate(words_list)
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(unique_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[51]:


# Abusive data after text processing
abusive = mail[mail["Class"]=="Abusive"]
abusive.head()


# In[52]:


# Abusive word cloud
abusive1 = ' '.join(abusive["final"])


# In[53]:


abusive_words = abusive1.split()


# In[54]:


ab_words  = [word for word in abusive_words if word not in add_stop_words]
ab_words = ' '.join(ab_words)


# In[55]:


abusive_wordcloud = WordCloud(width=600, height=400, stopwords="english").generate(ab_words)
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(abusive_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[56]:


# Non-abusive data after text processing
non_abusive = mail[mail["Class"]=="Non Abusive"]
non_abusive.head()


# In[57]:


# Non-abusive word cloud
non_abusive2 = ' '.join(non_abusive["final"])


# In[58]:


non_abusive_words = non_abusive2.split()


# In[59]:


non_ab_words  = [word for word in non_abusive_words if word not in add_stop_words]
non_ab_words = ' '.join(non_ab_words)


# In[60]:


non_abusive_wordcloud = WordCloud(width=600, height=400, stopwords='english').generate(non_ab_words)
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(non_abusive_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[61]:


# Adding 'length', 'words_count' and 'polarity' columns to dataset
from textblob import TextBlob


# In[62]:


mail['length'] = mail["final"].apply(lambda x: len(x))
mail['word_count'] = mail['final'].apply(lambda x: len(x.split()))
mail['polarity'] = mail['final'].apply(lambda x: TextBlob(x).sentiment.polarity)
mail.head()


# In[63]:


# Seperating the "Abusive" and "Non Abusive" classes
abusive = mail[mail["Class"]=="Abusive"]
abusive.head()


# In[64]:


abusive.shape


# In[65]:


non_abusive = mail[mail["Class"]=="Non Abusive"]
non_abusive.head()


# In[66]:


non_abusive.shape


# In[67]:


# Length distribution plot
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(abusive["length"], hist=True, label="abusive")
sns.distplot(non_abusive["length"], hist=True, label="non_abusive");


# In[68]:


sns.catplot(x="Class", y="length",data=mail, kind= "box")


# In[69]:


abusive.length.mean()


# In[70]:


non_abusive.length.mean()


# #### It can be observed that the mean length of non_abusive emails is higher than mean length of abusive emails

# In[71]:


# Word_count distribution plot
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(abusive.word_count, hist=True, label="abusive")
sns.distplot(non_abusive.word_count, hist=True, label="non_abusive");


# In[72]:


sns.catplot(x="Class", y="word_count",data=mail, kind= "box")


# In[73]:


abusive['word_count'].mean()


# In[74]:


non_abusive['word_count'].mean()


# #### It can be observed that the mean word_count of non_abusive emails is slightly higher than mean word_count of abusive emails

# In[75]:


# Polarity distribution plot
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(abusive.polarity, hist=True, label="abusive")
sns.distplot(non_abusive.polarity, hist=True, label="non_abusive");


# In[76]:


sns.catplot(x="Class", y="polarity",data=mail, kind= "box")


# In[77]:


abusive['polarity'].mean()


# In[78]:


non_abusive['polarity'].mean()


# #### It can be observed that the mean polarity of non_abusive emails is almost similar to the mean polarity of abusive emails. Hence both the mean of absive and non_abusive emails shows 'neutral' polarity 

# ### Distribution of Unigram, Bigram and Trigram

# In[79]:


from sklearn.feature_extraction.text import CountVectorizer


# In[80]:


def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(1, 1), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# In[81]:


words = get_top_n_words(mail['final'], 20)
words


# ### Unigram

# In[82]:


df_unigram = pd.DataFrame(words, columns= ['Unigram', 'Frequency'])
df_unigram


# In[83]:


plt.figure(figsize=(15,5))
g= sns.barplot("Unigram",'Frequency', data=df_unigram, color="b")
g.set_xticklabels(df_unigram["Unigram"], rotation=30)
plt.show()


# ### Bigram

# In[84]:


def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(2,2), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# In[85]:


words = get_top_n_words(mail['final'], 20)
words


# In[86]:


df_bigram = pd.DataFrame(words, columns= ['Bigram', 'Frequency'])
df_bigram


# In[87]:


plt.figure(figsize=(18,5))
g= sns.barplot("Bigram",'Frequency', data=df_bigram, color="b")
g.set_xticklabels(df_bigram["Bigram"], rotation=30)
plt.show()


# ### Trigram

# In[88]:


def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(3,3), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# In[89]:


words = get_top_n_words(mail['final'], 20)
words


# In[90]:


df_trigram = pd.DataFrame(words, columns= ['Trigram', 'Frequency'])
df_trigram


# In[91]:


plt.figure(figsize=(18,5))
g= sns.barplot("Trigram",'Frequency', data=df_trigram, color="b")
g.set_xticklabels(df_trigram["Trigram"], rotation=30)
plt.show()


# In[92]:


corpus = mail['final'].tolist()


# In[93]:


corpus[0:10]


# ## Building of Bag of Words

# In[94]:


from sklearn.feature_extraction.text import CountVectorizer


# In[95]:


cv = CountVectorizer(max_features=5000)

pickle.dump(cv, open('tranform.pkl', 'wb'))

# In[96]:


cv.get_params()


# In[97]:


# Indepedent variables
X = cv.fit_transform(corpus).toarray()


# In[98]:


X.shape


# In[99]:


X[0:10]


# In[100]:


mail.head()


# In[101]:


# Dependent variable
class_values = pd.get_dummies(mail['Class'])
class_values = class_values.drop(columns="Non Abusive")
class_values = class_values.rename(columns={"Abusive":"Class"})


# In[102]:


y = class_values.values.ravel()


# In[103]:


# Splitting the train test data
from sklearn.model_selection import train_test_split


# In[104]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[105]:


print(X_train.shape,'\n',
     X_test.shape)


# In[106]:


# Bag of words
count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())


# In[107]:


count_df.head()


# ## Model Building

# ### Building Naive Bayes model

# In[108]:


from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()


# In[109]:


nb_classifier.fit(X_train, y_train)


# In[110]:


# Predicting the values
y_pred_train = nb_classifier.predict(X_train)
y_pred_test = nb_classifier.predict(X_test)


# In[111]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[112]:


confusion_matrix(y_train, y_pred_train)


# In[113]:


confusion_matrix(y_test, y_pred_test)


# In[114]:


# Checking accuracy, precision and recall
# For training
accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)


# In[115]:


print("Accuracy for train: ", accuracy)
print("Precision for train: ", precision)
print("Recall for train: ", recall)


# In[116]:


# Checking accuracy, precision and recall
# For testing
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)


# In[117]:


print("Accuracy for test: ", accuracy)
print("Precision for test: ", precision)
print("Recall for test: ", recall)


# #### Highest accuracy achieved with Naive Bayes is 95.32%

# ### Hyperparameter tuning of Naive Bayes model

# In[118]:


# Tuning the parameter 'alpha' to improve the accuracy
classifier=MultinomialNB(alpha=0.1)


# In[119]:


previous_score=0
for alpha in np.arange(0.1,1.1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(round(alpha,1),score))


# #### Highest accuracy is 95.03% obtanined for the value alpha =0.1

# ### Logistic regression model

# In[120]:


from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(max_iter=500, random_state=0)


# In[121]:


lr_classifier.fit(X_train, y_train)


# In[122]:


# Predicting the values
y_pred_train = lr_classifier.predict(X_train)
y_pred_test = lr_classifier.predict(X_test)


# In[123]:


# Confusion matrix
confusion_matrix(y_train, y_pred_train)


# In[124]:


confusion_matrix(y_test, y_pred_test)


# In[125]:


# Accuracy score, precision and recall
# For training
Accuracy = accuracy_score(y_train, y_pred_train)
Precision = precision_score(y_train, y_pred_train)
Recall = recall_score(y_train, y_pred_train)
print("Accuracy for train: ", Accuracy)
print("Precision for train: ", Precision)
print("Recall for train: ", Recall)


# In[126]:


# Accuracy score, precision and recall
# For testing
Accuracy = accuracy_score(y_test, y_pred_test)
Precision = precision_score(y_test, y_pred_test)
Recall = recall_score(y_test, y_pred_test)
print("Accuracy for test: ", Accuracy)
print("Precision for test: ", Precision)
print("Recall for test: ", Recall)


# #### Highest accuracy achieved with Logistic Regression is 97.72%

# ### Hyperparameter tuning for logistic regression

# In[127]:


classifier=LogisticRegression(C=1)


# In[128]:


previous_score=0
for i in np.arange(0.1,1.1,0.1):
    sub_classifier=LogisticRegression(max_iter=500, C=i, random_state=0)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("C: {}, Score : {}".format(round(i,1),score))


# #### Highest accuracy is 98.23% obtained for the value of C=1

# """***Random Forest Classifier***"""

# In[129]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=252)
clf.fit(X_train,y_train)

pred_rf = clf.predict(X_test)


# """*ROC AUC Curve*"""

# In[130]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_rf)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# """find out **n_estimators**(**no. of trees**)"""

# In[131]:


from sklearn.ensemble import RandomForestClassifier    # ROC curve shows that, at 100 no. of trees it gives max. accuracy
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, label='Train AUC')
line2, = plt.plot(n_estimators, test_results, label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


# """# **Results**"""

# In[132]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
print(confusion_matrix(y_test, pred_rf))
print('\n')
print(classification_report(y_test,pred_rf))

print('Accuracy of Random Forest Classi:', accuracy_score(pred_rf ,y_test)*int(100))    # 99.11%


# """### *Training & Testing Results...*"""

# In[133]:


clf.score(X_train,y_train)    # training Accuracy = 99.96%

clf.score(X_test,y_test)      # testing Accuracy = 99.13%

pred_train = clf.predict(X_train)    # train_prediction
pred_test  = clf.predict(X_test)     # test_prediction

x_axis_labels = ["predict_Non_Abusive","predict_Abusive"]       # labels for x-axis
y_axis_labels = ["actual_Non_Abusive","actual_Abusive"]     # labels for y-axis


# """*Heat_Map for train_matrix*"""

# In[134]:


train_matrix = confusion_matrix(y_train,pred_train)

h1 = sns.heatmap(train_matrix,annot=True, fmt="d",annot_kws={"size": 17}, cmap='viridis',cbar_kws={'label': 'My Colorbar','orientation': 'horizontal'}, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
h1.set_yticklabels(h1.get_yticklabels(), rotation=0)
h1.set_title('train_confusion_matrix')


# """*Heat_Map for test_matrix*"""

# In[135]:


test_matrix = confusion_matrix(y_test,pred_test)       # test confusion_matrix

h2 = sns.heatmap(test_matrix,annot=True, fmt="d",annot_kws={"size": 17}, cmap='viridis',cbar_kws={'label': 'My Colorbar','orientation': 'horizontal'}, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
h2.set_yticklabels(h1.get_yticklabels(), rotation=0)
h2.set_title('test_confusion_matrix')


# In[136]:


from sklearn.naive_bayes import MultinomialNB
NAb_model = MultinomialNB(alpha=0.5)
NAb_model.fit(X_train,y_train)

pred_stem = NAb_model.predict(X_test)

print (confusion_matrix(y_test,pred_stem))
print('\n')
print (classification_report(y_test,pred_stem))

print('Model accuracy score with stemming :',accuracy_score(pred_stem, y_test) *int(100))   # 95.99%


# """SVM Algorithm"""

# In[137]:


from sklearn.svm import SVC

svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test,y_pred))

print('Accuarcy of SVM_Classi:', accuracy_score(y_pred,y_test)*int(100))    # 98.50%


# """Decision Tree Classifier"""

# In[138]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(min_samples_split=5, random_state=252)
dtc.fit(X_train,y_train)

pred_dtc = dtc.predict(X_test)

print(confusion_matrix(y_test,pred_dtc))
print('\n')
print(classification_report(y_test,pred_dtc))

print('Accuracy Decision_Tree_Classi:', accuracy_score(y_test,pred_dtc)*int(100))   # 98.80


# """Bagging Classifier"""

# In[139]:


from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier(n_estimators=20, random_state=252)
bc.fit(X_train,y_train)

pred_bc = bc.predict(X_test)

print(confusion_matrix(y_test,pred_bc))
print('\n')
print(classification_report(y_test,pred_bc))

print('Accuracy Bagging_Classi:', accuracy_score(y_test,pred_bc)*int(100))   # 99.00%


# In[140]:


filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

#filename='nlp_model.pkl'
#joblib.dump(clf,'nlp_model.pkl')
#joblib.dump(cv,'vector.pkl')


# In[ ]:


# filename='nlp_model.pkl'
# pickle.dump(clf,open(filename,'wb'))
# pickle.dump(cv,open('vector.pkl','wb'))


# In[ ]:




