
# -*- coding: utf-8 -*-
import sys
import os
import pytest


from ...datatools.datatools import *

def test_shape_of_dataset():
    pod_list = []
    for i in range(5):
        pod = DataPod(np.random.randn(1024,3),fs =1024)
    pod_list.append(pod)
    for i in range(5):
        pod = DataPod(np.random.randn(1025,3),fs =1025)
    pod_list.append(pod)
    _set = DataSet(pod_list)

    # We need to raise an error when this happens


    