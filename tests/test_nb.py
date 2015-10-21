#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

from tpmrec.nb import NB
import math


class TestNb(unittest.TestCase):
    def setUp(self):
        super(TestNb, self).setUp()
        self.x = ["He is a good boy",
                  "This is a pen",
                  "You have to pay money",
                  "Please give me money"]
        self.y = [0, 0, 1, 1]
        self.nb = NB()

    def test_train(self):
        self.nb.train(self.x, self.y)
        self.assertEqual(set([0, 1]), self.nb.categories)
        self.assertEqual(set(['He', 'is', 'a', 'good', 'boy', 'This', 'pen', 'You', 'have', 'to', 'pay', 'money', 'Please', 'give', 'me']), self.nb.vocabularies)
        self.assertEqual(2, self.nb.wordcount[0]['is'])
        self.assertEqual(2, self.nb.catcount[0])
        self.assertEqual(2, self.nb.catcount[1])

    def test_word_prob(self):
        self.nb.train(self.x, self.y)
        # calculate P(word|cat)
        # P(word='is'|cat=0) = (2 + 1) / (9 + 15) = 3/23
        self.assertEqual(3.0/24, self.nb.word_prob('is', 0))

    def test_score(self):
        self.nb.train(self.x, self.y)
        # calculate log(P(cat|doc))
        # log(p(cat=0|doc='He is')) ~= log(P(doc='He is'|cat=0)P(cat=0)) = log(2.0/24 * 3/24 * 2/4)
        self.assertEqual(math.log(2.0/24 * 3.0/24 * 2.0/4.0), self.nb.score('He is', 0))

    def test_predict(self):
        self.nb.train(self.x, self.y)
        self.assertEqual(0, self.nb.predict('He is a good boy'))
        self.assertEqual(1, self.nb.predict('You give me money'))


def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestNb))
    return suite


if __name__ == '__main__':
    unittest.main()
