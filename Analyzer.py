from tkinter import * 
from tkinter import filedialog
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

root=Tk()
files=[]
all_reviews=[]
data_train=[]
data_test=[]
#Functions
def addtrainfile():
    global data_train
    for widget in can.winfo_children():
        widget.destroy

    filename1=filedialog.askopenfilename(initialdir="/",title="Select Training Data",
                                         filetypes=(("Text","*.txt",),("all files","*.*")))
    print(filename1)    
    data_train=pd.read_csv(filename1, delimiter='\t', header=None)
    data_train.columns= ['Reviews_text', 'Review_Class']
    print(data_train)
    # prediction(data_train)
    return data_train

def addtestfile():
    for widget in can.winfo_children():
        widget.destroy

    filename2=filedialog.askopenfilename(initialdir="/",title="Select Test Data",
                                         filetypes=(("Text","*.txt",),("all files","*.*")))
    print(filename2)  
    data_test=pd.read_csv(filename2, delimiter='\t', header=None)
    data_test.columns= ['Reviews_text', 'Review_Class']
    print(data_test)
    return data_test





def clean_test_text(df):
    all_test = list()
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
        all_test.append(words)
    return all_test

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

def prediction():
    # data_imdb=pd.read_csv('imdb_labelled.txt', delimiter='\t', header=None)
    # data_imdb.columns= ['Reviews_text', 'Review_Class']

    # data_amazon = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
    # data_amazon.columns = ["Reviews_text", "Review_Class"]

    # data_yelp = pd.read_csv("yelp_labelled.txt", delimiter='\t', header=None)
    # data_yelp.columns = ["Reviews_text", "Review_Class"]

    # data = pd.concat([data_imdb, data_amazon, data_yelp])
    data=data_train
    all_reviews = clean_text(data)
    all_test=clean_test_text(data)
    TV = TfidfVectorizer(min_df=3)   
    X_train = TV.fit_transform(all_reviews).toarray()
    X_test=TV.fit_transform(all_test).toarray()
    y_train = data[["Review_Class"]]
    # print(np.shape(X))
    # print(np.shape(y))
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=4)
    # print(X_test.shape)
    classifier = svm.SVC(kernel='linear',gamma='auto', C=2)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    # reoprtprint= classification_report(y_test,y_predict)
    # print(classification_report(y_test,y_predict))
    # print(classification_report(y_test,y_predict))
    print(y_predict)
    Positives=np.count_nonzero(y_predict==1)
    Negative=np.count_nonzero(y_predict==0)
    if Negative>Positives:
        print("Kharap")
    else:
        print('Valo')
    











#Functions














#Ui Elements

can=Canvas(root,height=700,width=700,bg='#263D42')
can.pack()

frames= Frame(root,bg='#fff')
frames.place(relheight=0.8, relwidth=0.8,relx=0.1,rely=0.1)

report=Label(frames, text="reoprtprint")
report.pack()

Open_train= Button(root,text="Open Training data", 
                    padx=10,fg='white',bg='#263D42',command=addtrainfile) 
Open_train.pack()

Open_test= Button(root,text="Open test data", 
                    padx=10,fg='white',bg='#263D42',command=addtestfile) 
Open_test.pack()

Open_predictt= Button(root,text="Predict", 
                    padx=10,fg='white',bg='#263D42',command=prediction) 
Open_predictt.pack()

#Ui Elements

root.mainloop()