#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

from tpmrec.nb import NB

class TestNb(unittest.TestCase):
    def setUp(self):
        super(TestNb, self).setUp()

    def testEqual(self):
        self.assertEqual(1, 1) # this is just a sample (please remove asap)
        pass

if __name__ == '__main__':
        unittest.main()