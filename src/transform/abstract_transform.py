# -*- coding: utf-8 -*-
from __future__ import division, print_function


class AbstractTransform(object):
    def __init__(self, params):
        pass

    def __call__(self, sample):
        return sample
