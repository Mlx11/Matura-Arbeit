# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 21:12:08 2018

@author: Arbeiten
"""

li = []

try:
    import cvxopt.solvers
except:
    li.append("cvxopt")
    
try:
    import matplotlib.pyplot as plt
except:
    li.append("matplotlib.pyplot")
try:
    from sklearn import datasets
    import sklearn.model_selection
    from sklearn.model_selection import train_test_split
    import sklearn.model_selection
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except:
    li.append("sklearn")

try:
    import mahotas
except:
    li.append("mahotas")
    
try:
    import operator
except:
    li.append("operator")
    
try:
    import pandas as pd
except:
    li.append("pandas")
    
try:
    import  numpy as np
except:
    li.append("numpy")
    
try:
    import cv2
except:
    li.append("cv2 (opencv)")
    
try:
    import pickle
except:
    li.append("pickle")
    
try:
    import os
except:
    li.append("os")
  
try:
    import tkinter as tk 
except:
    li.append("tkinter")
  
try:
    import math
except:
    li.append("math")
  
files = []
try:
    from SVMs import MultiSVM, SVM
except:
    files.append("SVMs.py")
  
try:
    from Kernels import *
except:
    files.append("Kernels.py")
  
try:
    from Deskriptor import *
except:
    files.append("Deskriptor.py")
    

  
def popUp(text):   
    try:
        window = tk.Tk()
        window.wm_title("Import Test Raport")
        label = tk.Label(window, text=text)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(window, text="beenden", command=window.destroy)
        button.pack()
        window.mainloop()
    except:
        print(text)
       
if len(li) == 0 and len(files) == 0:
    popUp("Alle Bibliotheken und Dateien richtig geladen!")
else:   
    msg = ""
    if len(li) != 0:
        msg = "Folgende Bibliotheken wurden nicht richtig geladen: \n"
        for bib in li:
            msg += " - " + bib + "\n"
    if len(files) != 0:
        msg += "\n Folgende Dateien wurden nicht gefunden: \n"
        for file in files:
            msg += " - " + file + "\n"
    popUp(msg)



