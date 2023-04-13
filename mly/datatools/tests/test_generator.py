# -*- coding: utf-8 -*-
import sys
import os
import pytest

#from numpy.random import randint as rint
import numpy as np
from ...datatools.generator import *




def test_simple():

    np.random.seed(150914)

    dataset = generator(duration =1, fs =1024, size = 1, detectors = 'HLV')

    # TEST 1
    for pod in dataset:
        print(int(np.mean(pod.strain)*1e10))
        assert int(np.mean(pod.strain)*1e10) == 2237318
        assert dataset.exportData().shape == (1, 3, 1024)



    dataset = generator(duration =1, fs =1024, size = 2, detectors = 'HLV')

    # TEST 2
    for i, pod in enumerate(dataset):
        print(int(np.mean(pod.strain)*1e10))
        assert int(np.mean(pod.strain)*1e10) == [13598079, -27487272][i]
        assert dataset.exportData().shape == (2, 3, 1024)


# class GeneratorArguments:

#     def __init__(self,duration,fs,size,detectors):
#         self.duration = duration
#         self.fs = fs
#         self.size = size
#         self.detectors = detectors

#     def as_dict(self):
#         return dict(duration = self.duration
#                 ,fs = self.fs
#                 ,size = self.size
#                 ,detectors = self.detectors
#                 )
    
#     def as_list(self):
#         return [self.duration
#                 ,self.fs
#                 ,self.size
#                 ,self.detectors]

# class GeneratorKeywordArguments:

#     def __init__(self,kwargs):
#         self.kwargs = kwargs

#     def as_dict(self):
#         return self.kwargs

# # @pytest.fixture
# # def keyargs():
# #     _ = GeneratorKeywordArguments({})
# #     return _.as_dict()


# arguments = [dict(duration = rint(1,10)
#                   , fs = 2**rint(6,12)
#                   , size = rint(1,10)
#                   , detectors = 'HLV') for i in range(20)]


# keywords = [dict(labels = {'ee':121}
#                 ,backgroundType = 'optimal'
#                 ,injectionSNR = 0
#                 ,noiseSourceFile = None
#                 ,windowSize = rint(16,34) #(32)            
#                 ,timeSlides = None) for i in range(20)]

# @pytest.mark.parametrize("args,keyargs",[(a ,k) for a,k in zip(arguments, keywords) ])
# def test_generator_with_optimal_noise(args, keyargs, _shape = None):
#     parameters = dict(**args, **keyargs)
#     args = list(args.values())
#     print(args)
#     print(keyargs)
#     dataset = generator(*args, **keyargs)
#     print('testing shape')
#     if _shape is None:
#         _shape = (len(parameters['detectors']), int( parameters['duration'] * parameters['fs']))
#     assert dataset[0].shape == _shape

 




