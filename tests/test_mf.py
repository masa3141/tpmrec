#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

from tpmrec.mf import MF
import math


class TestMf(unittest.TestCase):
    def setUp(self):
        super(TestMf, self).setUp()
        pass

    def test_train(self):
        pass

    def test_predict(self):
        pass

    def test_rmse(self):
        pass


def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestMf))
    return suite


if __name__ == '__main__':
    unittest.main()
