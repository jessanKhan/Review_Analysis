import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tkinter import * 

root=Tk()
# myLebel=Label(root, text="Hello")
# C = Tk.canvas(root, bg="blue", height=400, width=400)
# myLebel.pack()
# can=Canvas(root,height=700,width=700,bg='#263D42')
# can.pack()

# Open_train= Button(root,text="Open Training data", padx=10,fg='white',bg='#263D42') 
# Open_train.pack()



data_imdb=pd.read_csv('imdb_labelled.txt', delimiter='\t', header=None)
data_imdb.columns= ['Reviews_text', 'Review_Class']

data_amazon = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
data_amazon.columns = ["Reviews_text", "Review_Class"]

data_yelp = pd.read_csv("yelp_labelled.txt", delimiter='\t', header=None)
data_yelp.columns = ["Reviews_text", "Review_Class"]

data = pd.concat([data_imdb, data_amazon, data_yelp])



def clean_text(df):
    all_reviews = list()
    lines = df["Reviews_text"].values.tolist()
   
    for text in lines:
        text = text.lower()
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        tokens = word_tokenize(text)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        PS = PorterStemmer()
        words = [PS.stem(w) for w in words if not w in stop_words]
        words = ' '.join(words)
        all_reviews.append(words)
    return all_reviews

all_reviews = clean_text(data)
# all_reviews[0:20]
#print(all_reviews[0:5])


TV = TfidfVectorizer(min_df=3)   
X = TV.fit_transform(all_reviews).toarray()
y = data[["Review_Class"]]
print(np.shape(X))
print(np.shape(y))

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=4)
# print(X_test.shape)

classifier = svm.SVC(kernel='linear',gamma='auto', C=2)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)

# print(classification_report(y_test,y_predict))
# print(y_predict)

Positives=np.count_nonzero(y_predict==1)
Negative=np.count_nonzero(y_predict==0)

if Negative>Positives:
    print("Kharap")
else:
    print('Valo')

root.mainloop()