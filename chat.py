#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:41:29 2020

@author: paolodrago
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')
import numpy
import tflearn
import tensorflow
import random
import json
#import requests
import pickle
#response = requests.get("https://jsonplaceholder.typicode.com/todos/1")
#todos = json.loads(response.text)
#print(todos["userId"])

test=[]
with open("intents.json") as read_file:
    data = json.load(read_file)
    
#saving our data so it does not need to run every time we use the model
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

#Loading in our words individually using nltk and a few for loops :)

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            all_words = nltk.word_tokenize(pattern)
            words.extend(all_words)
            docs_x.append(all_words)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
                labels.append(intent["tag"])
    
    #organzing our words as non-repetitive and lower case sing nltk"
    
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    
    #"training and testing output"
    
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        single_word = [stemmer.stem(w.lower()) for w in doc]
        
        for w in words: 
            if w in single_word:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
    
        training.append(bag)
        output.append(output_row)
    
    #changes my list into Arrays using the numpy library
    
    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
        
#developing the model

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

#DNN is a type of neural netword model

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
    
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

#function building for our bot to return sentences

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat():
    print("Start Talking... type quit to stop")
    while True:
        inp = input("You: ")
        if inp == "quit":
            break
        #feed our word function
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        print(random.choice(responses))
    
chat()
    
    
    
    
    
    
    
    
    
    
    
    
