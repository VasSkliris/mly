# -*- coding: utf-8 -*-
import sys
import os
import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from ...datatools.generator import *
from ...waveforms import csg


@pytest.fixture
def csg100_1():
    inj = csg(100, 1 , fs =1024)
    print(np.mean(inj))
    return inj

@pytest.fixture
def csg300_01():
    inj = csg(300, 0.1 , fs =1024)
    print(np.mean(inj))
    return inj

@pytest.fixture
def waveform_example_1(csg100_1):
    return csg100_1

@pytest.fixture
def waveform_example_2(csg300_01):
    return csg100_1

@pytest.fixture
def waveform_examples(csg100_1, csg300_01):
    return [csg100_1, csg300_01]



def test_simple():

    np.random.seed(150914)

    dataset = generator(duration =1, fs =1024, size = 1, detectors = 'HLV',shuffle = False)

    # TEST 1
    for pod in dataset:
        assert_almost_equal(np.mean(pod.strain),0.00022373184784334004)
        assert dataset.exportData().shape == (1, 3, 1024)


    dataset = generator(duration =1, fs =1024, size = 2, detectors = 'HLV',shuffle = False)

    # TEST 2
    for i, pod in enumerate(dataset):
        assert_almost_equal(np.mean(pod.strain),[-0.002748727264217992
                                                , 0.0013598079253561302 ][i])
        assert dataset.exportData().shape == (2, 3, 1024)

# def test_inj(csg100_1):

#     assert_almost_equal(np.mean(pod.strain),-1091)




# def test_injection_initialization_oldtxt(tmp_path_factory, csg300_01, detectors = 'HLV'):

#     np.random.seed(150914)

#     inj_directory = tmp_path_factory.mktemp("my_injection", numbered=False)
#     print(str(inj_directory))
    
#     for det in detectors:
        
#         inj_subdirectory = tmp_path_factory.mktemp(det, numbered=False)
#         file_path = inj_subdirectory / 'test_injection_01.txt'
#         injection = csg300_01
#         np.savetxt(str(file_path) , injection)


#     dataset = generator(duration =1, fs =1024, size = 1, detectors = 'HLV', injection_source = str(inj_directory) ,shuffle = False)


#     _, inj_type = injection_initialization(injection_source, detectors)

#     assert inj_type == 'oldtxt'









# def test_create_file(tmp_path):

#     d = tmp_path / "sub"
#     d.mkdir()
#     sd = tmp_path / d / "sub"
#     p = sd / "hello.txt"
#     p.write_text("lalalala")
#     print(list(d.iterdir()))
#     print(list(sd.iterdir()))
#     print(list(p.iterdir()))

#     assert p.read_text() == "lalalala"
#     assert len(list(tmp_path.iterdir())) == 1
#     assert 0==1




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

def test_create_file(tmp_path):

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("lalalala")
    assert p.read_text() == "lalalala"
    assert len(list(tmp_path.iterdir())) == 1




# def test_injection_initialization_DataPod() 