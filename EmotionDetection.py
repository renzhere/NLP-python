"""
Author: Reenu Mohandas
Student Number: R00171231
File: Emotion Detection
"""

import numpy as np
import pandas as pd
from sklearn import naive_bayes
from sklearn import metrics
from preProcessing import *
import csv
import sys
import nltk
import codecs
from nltk import word_tokenize
import re
import imp


# Files
# OUTPUT_FILE = "lexicons/lexicons_compiled.csv"

# Initial variables

max_values = {}
data = []
anger_count = 0
sad_count = 0
happy_count = 0
others_count = 0
pos_count = 0
neg_count = 0
sentence_type = []

# Read the data file
data1 = open('C:\\MScAI\\NaturalLanguageProcessing\\project2\\train_sample.txt', 'r', encoding='utf-8')
data2 = open('C:\\MScAI\\NaturalLanguageProcessing\\project2\\sample_w_o_label.txt', 'r', encoding='utf-8')
data = open('C:\\MScAI\\NaturalLanguageProcessing\\project2\\newDatasets\\newDatasets\\newTrain.txt', 'r', encoding="utf8")

#Preprocessing the lines. Read the lines. split then using \t.
lines = data1.readlines()
tuples_in_lines = list(map(lambda item: item.split("\t"), lines))

#Separate the Turn1, Turn2, Turn3. Merge them into a long sentence
sentences = list(map(lambda item: item[1]+" "+item[2]+" "+item[3], tuples_in_lines))

# Emo_List = ["anger", "sadness", "joy", "others"]
# Senti_List = ["positive", "negative"]



#***************************************************
# functions
#****************************************************
# taking the word category count, sorting it in reverse and returning the sorted array
# def sort_count_Dict(count_dict):
#     # aux = [(count_dict[key], key) for key in count_dict]
#     for key in count_dict:
#         aux= count_dict[key]
#     aux.sort()
#     aux.reverse()
#     return np.argsort(aux, axis=1)


# classifier
def N_B_Classifier(x_traindata, y_traindata, x_testdata, y_testdata):
    clf = naive_bayes.GaussianNB()
    clf.fit(x_traindata, y_traindata)
    prediction_NB = clf.predict(x_testdata)
    accuracy_of_prediction = metrics.accuracy_score(y_testdata, prediction_NB)
    return accuracy_of_prediction


# normalising the column into the range of 0 to 1
def scale_linear_bycolumn(x_train_data,high=1, low =0):
    mins = np.min(x_train_data)
    maxs = np.max(x_train_data)
    rng = maxs - mins
    return high - (((high - low) * (maxs - x_train_data)) / rng)

# NRC Emotion Lexicon: http://www.saifmohammad.com/WebPages/lexicons.html
#   Format: aback \t anger \t 0
# EMOLEX_FILE = r"C:\\MScAI\\NaturalLanguageProcessing\\project2\\NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
# match_count = 0
# add_count = 0

f1 = "C:\\MScAI\\NaturalLanguageProcessing\\project2\\sample NRC.txt"
f = "C:\\MScAI\\NaturalLanguageProcessing\\project2\\NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
# df= pd.read_csv(f1)
# words = df['_word']
# print(words)

nrc_data = pd.read_csv('NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', names=['word', 'category', 'association'], encoding='utf-8', sep='\t')
nrc_data = nrc_data[nrc_data.association == 1]

# print(nrc_data.head())
# print(nrc_data[nrc_data.word == 'charitable'].category.values.tolist())
detects = []
Emotion= []
class_data =[]
for item in sentences:
    words_in_sent = re.split(r'\W+', item)

    for item in words_in_sent:
        detects = []
        # print(item)
        # print(nrc_data[nrc_data.word == item].category.values.tolist())
        detects.append(nrc_data[nrc_data.word == item].category.values.tolist())
        for item in detects:
            for element in item:
                if any("anger" in element for element in item):
                    anger_count += 1
                    word_Emo = dict.update({"anger": anger_count})
                if any("sadness" in element for element in item):
                    sad_count += 1
                    word_Emo = dict.update({"sadness": sad_count})
                if any("joy" in element for element in item):
                    happy_count += 1
                    word_Emo = dict.update({"joy": happy_count})
                elif any(("anger" or "sadness" or "joy ") not in element for element in item) :
                    others_count += 1
                    word_Emo = dict.update({"others": others_count})

        # print(max(anger_count, sad_count, happy_count, others_count))
        # print("hello")
    count_list = (anger_count, sad_count, happy_count, others_count)
    sent_Emo_type = max(count_list)
    if sent_Emo_type == anger_count:
        Emotion.append('Angry')
        class_data.append(0)
        print("Angry Sentence")
    elif sent_Emo_type == sad_count:
        Emotion.append('Sad')
        class_data.append(1)
        print("Sad Sentence")
    elif sent_Emo_type == happy_count:
        Emotion.append('Happy')
        class_data.append(2)
        print("Happy Sentence")
    elif sent_Emo_type == others_count:
        Emotion.append('others')
        class_data.append(3)
        print("Others Type Sentence")

print(len(Emotion))
# length of file for newTrain.txt file is 29161

#For classification Accuracy prediction, we need to develop a classifier model which can
#be used to get the prediction results of a test dataset based of the Machine Learning model
# we have made. For this purpose, Naive Bayes classifier is used in this code.
#-------------------------------------------------------------------------------
#x-taindata is the training module used to train the Machine Learning model
#y_traindata is the class list obtained after training the Machine Learning model
# x_traindata = pd.read_csv("newTrain.txt", delimiter="\t", encoding='utf-8')
# y_traindata = Emotion

#-------------------------------------------------------------------------------
#x_testdata is the test module used to train the Machine Learning model
#y_traindata is the class list obtained after running the model on baseline model given
# x_testdata = pd.read_csv("newTestWithoutLabels.txt", delimiter="\t", encoding='utf-8')
# y_testdata = prediction_NB

#Normalising the values from the Machine Learning Module developed and NRC based module developed
#ML module normalised :
# ml_score_normalised = scale_linear_bycolumn(y_traindata)
# nrc_score_normalised = scale_linear_bycolumn(y_testdata)

# print('Naive Bayes Classifier: ', N_B_Classifier(x_traindata, y_traindata, x_testdata, y_testdata ))










