import pandas as pd
import numpy as np

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk import ngrams
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import TweetTokenizer
nltk.download('punkt')
nltk.download('stopwords')

# Предобработка данных
positive = pd.read_csv('positive.csv', sep=';', usecols=[3], names=['text'])
positive['label'] = ['positive'] * len(positive)

negative = pd.read_csv('negative.csv', sep=';', usecols=[3], names=['text'])
negative['label'] = ['negative'] * len(negative)

df = positive.append(negative)
x_train, x_test, y_train, y_test = train_test_split(df.text, df.label)

# Модель 1
smart_vectorizer = CountVectorizer(ngram_range=(2, 5))
smart_vectorized_x_train = smart_vectorizer.fit_transform(x_train)

clf = DecisionTreeClassifier(random_state=88,max_depth = 50)
clf.fit(smart_vectorized_x_train, y_train)

smart_vectorized_x_test = smart_vectorizer.transform(x_test)

pred = clf.predict(smart_vectorized_x_test)
print(classification_report(y_test, pred))

# Модель 2 (шум и токенизатор)
noise = stopwords.words('russian') + list(punctuation)
tt = TweetTokenizer()
smart_vectorizer = CountVectorizer(ngram_range=(2, 5),
                                  tokenizer = tt.tokenize,
                                   stop_words = noise)
smart_vectorized_x_train = smart_vectorizer.fit_transform(x_train)

clf = DecisionTreeClassifier(random_state=88,max_depth = 50)
clf.fit(smart_vectorized_x_train, y_train)

smart_vectorized_x_test = smart_vectorizer.transform(x_test)

pred = clf.predict(smart_vectorized_x_test)
print(classification_report(y_test, pred))

# Модель 3 (токенизатор)
smart_vectorizer = CountVectorizer(ngram_range=(2, 5),
                                   tokenizer = tt.tokenize)
smart_vectorized_x_train = smart_vectorizer.fit_transform(x_train)

clf = DecisionTreeClassifier(random_state=88,max_depth = 50)
clf.fit(smart_vectorized_x_train, y_train)

smart_vectorized_x_test = smart_vectorizer.transform(x_test)

pred = clf.predict(smart_vectorized_x_test)
print(classification_report(y_test, pred))

# Модель 4 (n_gramm = (1,5))

smart_vectorizer = CountVectorizer(ngram_range=(1, 5),
                                   tokenizer = tt.tokenize)
smart_vectorized_x_train = smart_vectorizer.fit_transform(x_train)

clf = DecisionTreeClassifier(random_state=88,max_depth = 50)
clf.fit(smart_vectorized_x_train, y_train)

smart_vectorized_x_test = smart_vectorizer.transform(x_test)

pred = clf.predict(smart_vectorized_x_test)
print(classification_report(y_test, pred))
