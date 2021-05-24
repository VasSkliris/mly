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

"""Tests for the `mly` top-level module.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

import os

import mly


def test_CreateMLyWorkbench(tmp_path):
    """Test the `mly.CreateMLyWorkbench` function
    """
    # move into the temporary directory
    oldcwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # call the function
        mly.CreateMLyWorkbench()
        # check that the directories are created as expected
        for dirn in (
            "datasets",
            "trainings",
            "injections",
        ):
            assert (tmp_path / "MLy_Workbench" / dirn).is_dir()
    finally:
        # in all cases (pass/fail) return to where we started
        # (if we don't do this manually, something bad will probably happen when
        #  pytest attempts to clean up the tmp_path)
        os.chdir(oldcwd)
