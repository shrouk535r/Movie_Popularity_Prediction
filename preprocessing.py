import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import re
import string
from textblob import Word
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
stopwords = set(stopwords.words('english'))

import matplotlib.pyplot as plt
import seaborn as sns
import bisect
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sqlalchemy.sql.expression import column
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
stopwords = set(stopwords.words('english'))
lbl = LabelEncoder()
class preprocessing:
    #fill null
    def process(self,X_train,X_test):
        X_train = self.handle_hompage(X_train)
        X_test = self.handle_hompage(X_test)
        X_train = self.removedupli(X_train)
        X_test = self.removedupli(X_test)
        X_train = self.handle_date(X_train)
        X_test = self.handle_date(X_test)

        LS = list()

        dol = ['genres', 'keywords', 'production_companies', 'production_countries','spoken_languages']

        selected_list=['name','name','name','name','iso_639_1']
        for i in range(len(dol)):
            X_train = self.Loctolist(X_train, dol[i], selected_list[i])
            X_test = self.Loctolist(X_test, dol[i], selected_list[i])
            ls = self.fit_List(X_train, dol[i])
            #here ls
            X_train = self.transform_List(X_train, dol[i], ls)
            X_test = self.transform_List(X_test, dol[i], ls)

            # SECOND_WAY

        # SECOND_WAY
        # X_train,X_test=self.encod(X_train,X_test,ls)
        # print(X_train['original_language'])
        # print(X_test['original_language'])
        X_train, col = self.drop_coloums(X_train)
        #here col
        X_test=self.drop_columns_test(X_test,col)
        X_train,means = self.handlenull(X_train)
        X_test = self.fill_null(X_test,means)
        print(X_train.info())
        print(X_test.info())
        x_tre = self.preprocc(X_train['overview'])
        X_tes = self.preprocc(X_test['overview'])
        x_train_overview, vect = self.tf_ifd(x_tre)
        x_test_overview = self.tf_idf_transform(X_tes, vect)
        for i in vect.get_feature_names():
            X_train[i]=x_train_overview[i]
            X_test[i]=x_test_overview[i]
        # X_train = pd.concat([X_train, x_train_overview], axis=1,join='inner')
        # X_test = pd.concat([X_test, x_test_overview], axis=1,join='inner')
        X_train = X_train.drop('overview', axis=1)
        X_test = X_test.drop('overview', axis=1)


        #
        # text = X_train['overview']
        # text1=X_test['overview']
        # nltk.download('punkt')
        # nltk.download('stopwords')
        # nltk.download('wordnet')
        # textt = text.apply(self.prepare)
        # textt1 = text1.apply(self.prepare)
        # X_train['cleaned_overview'] = textt.apply(self.clean_overview_text)
        # X_test['cleaned_overview'] = textt1.apply(self.clean_overview_text)
        # print(X_train['cleaned_overview'])
        # print(X_test['cleaned_overview'])
        # X_train = X_train.drop('overview', axis=1)
        # X_test = X_test.drop('overview', axis=1)


        print(X_train.info())
        print(X_test.info())
        # SECOND_WAY
       # print(X_train.shape(),X_test.shape())
        # X_train, X_test,lab = self.encod(X_train, X_test, LS)
        self.convert(X_train)
        X_train=self.convert_transform(X_train)
        X_test=self.convert_transform(X_test)
        X_train=self.one_hot_encode(X_train)
        X_test=self.one_hot_encode(X_test)

        #here lab
        colm = X_train.columns
        # print(X_train.info())
        # print(X_test.info())
        scalar = self.feature_scaling(X_train)
        #here scalar
        X_train=self.transform_scaling(X_train,scalar)
        X_test=self.transform_scaling(X_test,scalar)


        return X_train,X_test
    def most_freq(self, X, col):
        mostfreq = X[col].value_counts()

        mostfreq = mostfreq[:1].idxmax()
        #print(mostfreq)
        return mostfreq

    def handlenull(self, X):
        mean_values={}
        for i in X:
            if X[i].dtypes == 'int64' or X[i].dtypes == 'float64':
                mean_values[i]=X[i].median()
                X[i] = X[i].fillna(value=X[i].median())
            else:
                mostfreq_value = self.most_freq(X, i)
                mean_values[i]= mostfreq_value
                X[i] = X[i].fillna(value=mostfreq_value)

        return X,mean_values
    def fill_null(self,X,means):
        for i in X:
            X[i] = X[i].fillna(value=means[i])
        return X
    #label encoding
    #SECOND_WAY

    # def encod(self,Xtr, Xte, ls):
    #     cols = ls
    #     for c in cols:
    #         print(c)
    #         lbl = LabelEncoder()
    #         lbl.fit(list(Xtr[c].values))
    #         import bisect
    #         le_classes = lbl.classes_.tolist()
    #         print(lbl.classes_)
    #         for s in Xte[c]:
    #             if s not in le_classes:
    #                 bisect.insort_left(le_classes, s)
    #         lbl.classes_ = le_classes
    #         print(lbl.classes_)
    #         Xtr[c] = lbl.transform(list(Xtr[c].values))
    #         Xte[c] = lbl.transform(list(Xte[c].values))
    #     return Xtr,Xte# def encod(self,Xtr, Xte, ls):
    # def encod(self,Xtr, Xte, ls):
    #     cols = ls
    #     for c in cols:
    #         print(c)
    #         lbl = LabelEncoder()
    #         lbl.fit(list(Xtr[c].values))
    #         import bisect
    #         le_classes = lbl.classes_.tolist()
    #         print(lbl.classes_)
    #         Xte[c]=Xte[c].map(lambda s: '<unseen>' if s not in le_classes else s)
    #         bisect.insort_left(le_classes, '<unseen>')
    #         lbl.classes_ = le_classes
    #         print(lbl.classes_)
    #         Xtr[c] = lbl.transform(list(Xtr[c].values))
    #         Xte[c] = lbl.transform(list(Xte[c].values))
    #     return Xtr,Xte
    # def encode_test(self,x,ls,lbl):
    #
    #     cols = ls
    #     for c in cols:
    #         import bisect
    #         le_classes = lbl.classes_.tolist()
    #         print(lbl.classes_)
    #         x[c]=x[c].map(lambda s: '<unseen>' if s not in le_classes else s)
    #         bisect.insort_left(le_classes, '<unseen>')
    #         lbl.classes_ = le_classes
    #         print(lbl.classes_)
    #         x[c] = lbl.transform(list(x[c].values))
    #     return x
    def encod(self,Xtr, Xte, ls):
        cols = ls
        #print(cols)
    # def encod(self,Xtr, Xte):
    #     cols = ['status', 'original_language', 'tagline','title','original_title']
        x_new = pd.concat([Xtr, Xte], axis=0)
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(x_new[c].values))
            #Xte[c] = Xte[c].map(lambda s: '<unseen>' if s not in lbl.classes_ else s)
            Xtr[c] = lbl.transform(list(Xtr[c].values))
            Xte[c] = lbl.transform(list(Xte[c].values))
        return Xtr,Xte,lbl
    #preprocess list of dictionary
    def convert(self,X):
        labels = np.array(['English', 'other Languages'])
        en_count = X['original_language'].value_counts()[0]
        perc = np.array([en_count, sum(X['original_language'].value_counts()) - en_count])
        plt.figure(figsize=(7, 7))
        plt.pie(perc, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.show()
    def convert_transform(self,X):
        X['original_language'] = X['original_language'].map(lambda s: 'other Languages' if s !='en' else s)
        return X
    def one_hot_encode(self,X):
        # ls = ['status', 'original_language']
        # for i in ls:
        X['status'] = X['status'].apply(lambda x: 1 if 'Released' == x else 0)
        X['original_language'] = X['original_language'].apply(lambda x: 1 if 'en' == x else 0)
        return X

    #1- conver list of dictionary to list
    def Loctolist(self,X, c, selectlist):
        lsname = []
        for indexD in X.index:
            ListOFdictionay = json.loads(X[c][indexD])
            for i in range(len(ListOFdictionay)):
                lsname.append(ListOFdictionay[i][selectlist])
            X[c][indexD] = lsname
            lsname = []
        return X

    #2- convert list to one hot encoding(fit)
    #SECOND_WAY
    # def fit_List(self,X, c):
    #     ls2 = []
    #     for indexD in X.index:
    #         ListOFdictionay = X[c][indexD]
    #         ls2.append(len(ListOFdictionay))
    #     count=0
    #     rls=[]
    #     for red in reversed(list(set(ls2))):
    #         count+=ls2.count(red)
    #         if (count/len(X))*100 >=50:
    #             rls.append(red)
    #     n = max(rls)
    #     newl = list()
    #     for i in range(n):
    #         newl.append(c + str(i + 1))
    #     print(newl)
    #     print(len(newl))
    #     return newl
    def fit_List(self,X, c):
        ls = []
        # ls2 = []
        for indexD in X.index:
            ListOFdictionay = X[c][indexD]

            for i in range(len(ListOFdictionay)):
                ls.append(ListOFdictionay[i])
            # ls2.append(len(ListOFdictionay))
        newl = list(set(ls))
        rsult=list()
        for i in newl:
            x = ls.count(i)
            # print(i + " : " + str(x))
            #try columns
            if x > 100:
                rsult.append(i)
        print(rsult)
        print(len(rsult))
        return rsult

    #3- convert list to one hot encoding(transform)
    #SECOND_WAY
    # def transform_List(self,X, c, ls):
    #     for indexD in X.index:
    #         List_dictionay = X[c][indexD]
    #         for i in range(len(ls)):
    #             if i < len(List_dictionay):
    #                 X.at[indexD, ls[i]] = List_dictionay[i]
    #             else:
    #                 X.at[indexD, ls[i]]=""
    #     X = X.drop([c], axis=1)
    #     return X
    def transform_List(self,X, c, ls):

        for indexD in X.index:
            #List_dictionay = X.at[indexD, ls[i]]
            List_dictionay = X[c][indexD]

            for i in range(len(ls)):
                if ls[i] in List_dictionay:
                    X.at[indexD, ls[i]] = 1
                else:
                    X.at[indexD, ls[i]] = 0
        X = X.drop([c], axis=1)
        return X

    # scalling
    def feature_scaling(self,X_train):
        scaler = MinMaxScaler().fit(X_train)
        # revenue = data['revenue']
        # x = revenue.values.reshape(-1, 1)  # returns a numpy array
        # scaled_revenue = scaler.fit(x)

        return scaler;
    def transform_scaling(self,x,scaler):
        x = pd.DataFrame(scaler.transform(x), columns=x.columns, index=x.index)
        return x;
    def removedupli(self,X):
        if(X.duplicated().sum()>0):
            X=X.drop_duplicates()
        return X
    def handle_date(self,X):
        ralease_date = pd.DatetimeIndex(X['release_date'], dayfirst=False)
        X['Year'] = ralease_date.year
        X['Month'] = ralease_date.month
        X['Day'] = ralease_date.day
        X = X.drop(['release_date'], axis=1)
        return X

    def handle_hompage(self,X):
        print("precentage of null in homepage :"+str((X.homepage.isnull().sum().sum()/len(X.homepage))*100))
        X['is_homepage'] = np.where(X['homepage'].isnull(), 0, 1)
        # X = X.drop(['homepage'], axis=1)
        return X


    def correlation(self,data, threshold):
        col_corr = set()  # set of all the names of correlated columns
        corr_matrix = data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]  # getting the name of columns
                    col_corr.add(colname)
        #print(len(col_corr))
        #print(col_corr)
        return col_corr


    def feature_selection(self,x_train,y_train):
        # feature = self.correlation(x_train, 0.9)
        # x_train = x_train.drop(feature, axis=1)
        # x_test = x_test.drop(feature, axis=1)
        # plt.subplots(figsize=(12, 8))
        # top_corr = x_train.corr()
        # sns.heatmap(top_corr, annot=True)
        # plt.show()
        corr_matrix = pd.concat([x_train, y_train], axis=1).corr()
        corra = corr_matrix['vote_average'].sort_values(ascending=False)
        print(corra)
        top_feature = corr_matrix.index[abs(corr_matrix['vote_average']) > .1]
        print(top_feature)
        plt.subplots(figsize=(12, 14))
        top_corr = pd.concat([x_train, y_train], axis=1)[top_feature].corr()
        sns.heatmap(top_corr, annot=True)
        plt.show()
        top_feature=top_feature.delete(-1)
        return top_feature
    def feature_selection_transform(self,x,top_feature):
        x = x[top_feature]
        return x
    def wrapper_feature_selection(self,x_train,y_train):
        sfs = SFS(DecisionTreeClassifier(max_depth=6,random_state=10),
                  k_features=(13),
                  forward=True,
                  floating=False,
                  scoring='accuracy',
                  n_jobs=-1,
                  cv=0)
        SFS_results=sfs.fit(x_train, y_train)
        print(SFS_results.k_feature_names_)
        print(len(SFS_results.k_feature_names_))
        return SFS_results
        # Ordered_rank_feature = SelectKBest(score_func=chi2, k=30)
        #
        # Ordered_feature = Ordered_rank_feature.fit(x_train,y_train)
        # print("features",Ordered_feature)
    def wrapper_feature_selection_transform(self, x, SFS_results):
        x= pd.DataFrame(SFS_results.transform(x), columns=SFS_results.k_feature_names_,index=x.index)
        return x
    def calc_nulls(self,data):
        df=data.isnull().sum()
        null_percentage = (df / data.isnull().count() * 100)
        print("Null values percantage...")
        display_null_percent = pd.concat([df,null_percentage], axis=1, keys=['Total', 'Percent'])
        print(display_null_percent)
        print("Data Shape: %s\n" % (data.shape,))
        return null_percentage

    def drop_coloums(self,Xtrain):
        percentage=self.calc_nulls(Xtrain)
        col=list(['id','tagline', 'title', 'original_title', 'overview'])
        for i in col:
            print(i+" : "+ str(len(Xtrain[i].unique())/len(Xtrain[i])))
        Xtrain=Xtrain.drop(col,axis=1)
        for c in percentage.iteritems():
            if c[1] > 50:
                print(c[0]+':'+str(c[1])+'%')
                col.append(c[0])
                Xtrain = Xtrain.drop(c[0], axis=1)
        print("Train Data Shape after columns removal: %s" % (Xtrain.shape,))
        return Xtrain,col
    def drop_columns_test(self,x,col):
        for c in col:
            x = x.drop(c, axis=1)
            print(c)
        print("Test Data Shape after columns removal: %s" % (x.shape,))
        return x

    def preprocc(self,X):
        courpos = []
        for i in X:
            # print(file)
            text = i
            # 1- convert to lowerCase
            text = text.lower()
            # 2-REMOVE ANY SPEACIAL CHARACTER  AND NUMBERS
            text = re.sub("[^a-zA-Z]", " ", text)

            # 3-TOKANIZATION
            word_tokinzed = word_tokenize(text)

            # 4-LEMMATIZATION
            lemmatizer = WordNetLemmatizer()
            words = [Word(word).lemmatize() for word in word_tokinzed]
            st = PorterStemmer()

            word_stemmed = [st.stem(word) for word in word_tokinzed]
            words = [Word(word).lemmatize() for word in word_stemmed]

            # 5-Stop Words Removel
            filtered = [word for word in words if word.casefold() not in stopwords]

            # # Remove Punctuation strings
            # no_of_punc_words = [''.join(char for char in word if char not in string.punctuation)
            #                     for word in filtered]

            # #6-Remove the empty string
            # no_of_punc_words = [word for word in filtered if word]
            # print(len(no_of_punc_words))
            # word_counter = Counter(no_of_punc_words)

            # 6-JOIN THE TEXT
            no_of_punc_words = " ".join(filtered)

            # LIST OF TEXTS
            courpos.append(no_of_punc_words)
        courposArray = np.array(courpos)
        data = pd.DataFrame(courposArray)

        return data

    def tf_ifd(self,X_train):
        vectorizer = TfidfVectorizer(min_df=200, use_idf=True)
        vectors_train = vectorizer.fit_transform(X_train[0]).toarray()
        # print()
        print("vectors_train : ", vectors_train.shape)
        X_train = pd.DataFrame(vectors_train, columns=vectorizer.get_feature_names())
        return X_train, vectorizer

    def tf_idf_transform(self,X_test, vectorizer):
        vectors_test = vectorizer.transform(X_test[0]).toarray()
        X_test = pd.DataFrame(vectors_test, columns=vectorizer.get_feature_names())
        return X_test

    # load the dataset
    def prepare(self,text):
        if isinstance(text, str):
            # remove HTML tags
            text = BeautifulSoup(text, 'html.parser').get_text()
            return text
        else:
            return ''

    # cleaning the overview text
    def clean_overview_text(self,text):

        # convert to lowercase
        text = text.lower()

        # remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        # remove stop words
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stopwords]

        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # join words back into a string
        clean_text = ' '.join(words)
        return clean_text

    # apply the function to the overview column

    # def preprocessingwithnlp(self,column):