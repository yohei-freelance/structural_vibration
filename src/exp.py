#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:53:37 2019

@author: yohei
"""

import numpy as np

A = np.array([[1, 3],
              [-2, -4]])
print(np.linalg.eig(A))