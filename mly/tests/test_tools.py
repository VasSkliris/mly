# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2021)
#
# This file is part of MLY.
#
# MLY is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MLY is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MLY.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for the `mly.tools` module.
"""

__author__ = "Vasileios Skliris <vasileios.skliris@ligo.org>"

import os
import numpy as np
from mly.tools import *

# TEST for mly.tools.dirlist

cwd = os.getcwd()

def test_dirlist():
    assert dirlist(cwd)
    


# TEST for mly.tools.lcm

def test_lcm():
    assert lcm(2,3,4) == 12
    assert lcm(1,1,8) == 8
    assert lcm(7,11,9) == 693

# TEST for mly.tools.circularTimeSlides

def test_circularTimeSlides():
    
    assert circularTimeSlides(1,10)==[[0]]
    assert circularTimeSlides(2,11).shape == (11,2)
    assert circularTimeSlides(3,15).shape == (15,3)
    
# TEST for mly.tools.internalLags

def test_internalLags():

    trials=[['HV',2,90,'HLV',3,111,'HLVL',2,97]]
    
    for trial in trials:
        indeces=internalLags(trial[0],trial[1],trial[2],fs=2,start_from_sec=0,includeZeroLag=False)

        groups=[]   # combinations of all detectors
        pairDict={} # combination of each pair of detectors

        for j in range(len(indeces.keys())):
            for k in range(1+j,len(indeces.keys())):
                pairDict[list(indeces.keys())[j]+list(indeces.keys())[k]]=[]


        for i in range(len(indeces[list(indeces.keys())[0]])):

            # Testing if the same group of indeces appears more than once
            group=[]
            for key in indeces.keys():
                group.append(indeces[key][i])

            assert (group not in groups), "Duplicate in indeces among all detectors ,"+str(group)

            groups.append(group)

            # Testing if the same pair of indeces appears more than once 
            # in each pair of detectors
            for j in range(len(indeces.keys())):
                for k in range(1+j,len(indeces.keys())):
                    pair=[indeces[list(indeces.keys())[j]][i],indeces[list(indeces.keys())[k]][i]]
                    #print(j,k,pair)

                    assert (pair not in pairDict[list(indeces.keys())[j]
                            +list(indeces.keys())[k]]), "Duplicate in indeces among a pair of detectors"
                    # In case we have same 
                    assert (pair[0]!=pair[1]), "zero-lag detected"

                    pairDict[list(indeces.keys())[j]+list(indeces.keys())[k]].append(pair)


# TEST for mly.tools.toCategorical

def test_toCategorical():
    labels=['a','a','a','a','b','b','b','b']
    mapping_translation={'a':[0,1],'b':[1,0]}
    
    try:
        result = toCategorical(labels,translation=False,from_mapping=mapping_translation)
    except TypeError:
        print("fromCategorical: Passed type test")
    
    mapping=['b','a']

    result = toCategorical(labels,translation=False,from_mapping=mapping)
    assert isinstance(result,(list,np.ndarray))
    for el in result:
        assert (1 in el) and (0 in el)
    assert (result[0]==np.array([0,1])).all()
    assert (result[-1]==np.array([1,0])).all()
        
    result = toCategorical(labels,translation=True,from_mapping=mapping)
    
    assert isinstance(result[0],(list,np.ndarray))
    
    for el in result[0]:
        assert (1 in el) and (0 in el)
        
    assert result[1]==mapping_translation
        
def test_correlate():
    
    x=np.random.randn(1024)
    y=np.random.randn(1024)
    
    assert len(correlate(x,y,4)) == 8
    assert len(correlate(x,y,7)) == 14

    

# def test_fromCategorical():
#     labels=['a','a','a','a','b','b','b','b']
#     mapping=['b','a']
#     categorical,translation=toCategorical(labels
#                                           ,translation=True
#                                           ,from_mapping=mapping)
#     print(translation)
#     print(categorical[0])
#     result=fromCategorical(categorical[0],mapping=translation)
    
#     assert result==labels
    
    
    
# def test_CreateMLyWorkbench(tmp_path):
#     """Test the `mly.CreateMLyWorkbench` function
#     """
#     # move into the temporary directory
#     oldcwd = os.getcwd()
#     os.chdir(tmp_path)
#     try:
#         # call the function
#         mly.CreateMLyWorkbench()
#         # check that the directories are created as expected
#         for dirn in (
#             "datasets",
#             "trainings",
#             "injections",
#         ):
#             assert (tmp_path / "MLy_Workbench" / dirn).is_dir()
#     finally:
#         # in alpytel cases (pass/fail) return to where we started
#         # (if we don't do this manually, something bad will probably happen when
#         #  pytest attempts to clean up the tmp_path)
#         os.chdir(oldcwd)
