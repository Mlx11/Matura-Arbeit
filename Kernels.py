# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:02:19 2018

@author: Arbeiten
"""

import numpy as np
import math

class Kernel: 
    def __init__(self):
        print("Kernel")
    
    
class LinearKernel(Kernel):
    def __init__(self):
        #print("init LinearKErnel")
        pass
    
    def calculate(self, x1,x2):
        return np.dot(x1,x2)
    
    def __str__(self):
        return "Linear Kernel"
    

class PolyKernel(Kernel):
    def __init__(self, p=2, c=0):
        #print("init poly kernel")
        self.p = p
        self.c = c
    
    def calculate(self, x1,x2):
        
        return pow(np.dot(x1,x2)+self.c, self.p)
    
    def __str__(self):
        return "Poly Kernel, p="+str(self.p)+", c="+str(self.c)
    
class RBFKernel(Kernel):
    def __init__(self, c=1):
        #print("init RBF kernel")
        self.c = c
    
    def calculate(self, x1,x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        norm = np.linalg.norm(x1-x2)
        return math.exp(-1*self.c*norm)
    
    def __str__(self):
        return "RBF Kernel"
    
    
    
    
class MyKernel1(Kernel):
    def __init__(self):
        #print("init LinearKErnel")
        pass
    
    def calculate(self, x1,x2):
        return np.dot(x1,x2)  + np.dot(x1,x2)**2
    
    def __str__(self):
        return "MyKernel1"
    
    
class MyKernel2(Kernel):
    def __init__(self):
        #print("init LinearKErnel")
        pass
    
    def calculate(self, x1,x2):
        return np.dot(x1,x2) + (np.dot(x1,x2)+1)**2
    
    def __str__(self):
        return "MyKernel2"
    
class MyKernel3(Kernel):
    def __init__(self):
        #print("init LinearKErnel")
        pass
    
    def calculate(self, x1,x2):
        t = float((np.dot(x1,x2)+1)**2)
        return float(np.dot(x1,x2)) * t
    
    def __str__(self):
        return "MyKernel3"
    
    
    
    
    
    
    
    
    





