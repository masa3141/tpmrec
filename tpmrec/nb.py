#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Naive bayes

You can run followed codes.
nb = NB()
nb.train(input_data)
nb.predict(test_data)
"""

import numpy as np
from collections import defaultdict
import math


class NB():
    """
    NB()
    """

    def __init__(self):
        """ init paramaters """
        self.categories = set()
        self.vocabularies = set()
        self.wordcount = {}
        self.catcount = defaultdict(int)

    def train(self, input_data, label_data):
        """ train model """
        # count category and word
        for i in range(len(input_data)):
            cat, doc = label_data[i], input_data[i].split(' ')
            self.categories.add(cat)
            self.wordcount.setdefault(cat, defaultdict(int))
            self.catcount[cat] += 1
            for word in doc:
                self.vocabularies.add(word)
                self.wordcount[cat][word] += 1

    def word_prob(self, word, cat):
        """ calculate P(word|cat) """
        return float(self.wordcount[cat][word] + 1) / float(sum(self.wordcount[cat].values()) + len(self.vocabularies))

    def score(self, doc, cat):
        """ calculate log(P(cat|doc)) """
        numdoc = sum(self.catcount.values())  # number of documents
        score = math.log(float(self.catcount[cat]) / numdoc)  # log P(cat)
        for word in doc.split(' '):
            score += math.log(self.word_prob(word, cat))
        return score

    def predict(self, doc):
        """ return the suitable category of document """
        best_cat = None
        max_score = -10000
        for cat in self.catcount.keys():
            p = self.score(doc, cat)
            if p > max_score:
                max_score = p
                best_cat = cat
        return best_cat

if __name__ == "__main__":
    x_input = ["He is a good boy",
               "This is a pen",
               "You have to pay money",
               "Please give me money"]
    y_label = [0, 0, 1, 1]

    nb = NB()
    nb.train(x_input, y_label)
    print "P(He|cat=0) = ", nb.word_prob("He", 0)
    print "P(He|cat=1) = ", nb.word_prob("He", 1)
    print "P(is|cat=0) = ", nb.word_prob("is", 0)
    print "P(is|cat=1) = ", nb.word_prob("is", 1)
    print "P(money|cat=0) = ", nb.word_prob("money", 0)
    print "P(money|cat=1) = ", nb.word_prob("money", 1)

    test_data = "This is a good pen"
    print "log P(cat=0|test_data) = ", nb.score(test_data, 0)
    print "log P(cat=1|test_data) = ", nb.score(test_data, 1)
    print "category of test_data = ", nb.predict(test_data)

    test_data = "I want money money money"
    print "log P(cat=0|test_data) = ", nb.score(test_data, 0)
    print "log P(cat=1|test_data) = ", nb.score(test_data, 1)
    print "category of test_data = ", nb.predict(test_data)
