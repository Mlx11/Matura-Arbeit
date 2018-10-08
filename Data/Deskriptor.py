# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:36:40 2018

@author: Arbeiten
"""

import cv2
import numpy as np
import mahotas
import os
import matplotlib.pyplot as plt
from SVMs import MultiSVM, SVM
from Kernels import LinearKernel, PolyKernel, RBFKernel, MyKernel1, MyKernel2, MyKernel3
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from multiprocessing import Pool
import time

class Deskriptor:
    #Deskriptoren aus https://github.com/Gogul09/image-classification-python
    
    #constants
    IMAGESIZE = tuple((100,100))
    BINS = 8
    
    def __init__(self, imgPath):
        img = cv2.imread(imgPath)
        self.img = cv2.resize(img, Deskriptor.IMAGESIZE)
        self.bwimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.vector = None
                
    
    def colhist(self):
        colhistogram = cv2.calcHist([cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)], [0,1,2], None, [Deskriptor.BINS, Deskriptor.BINS, Deskriptor.BINS], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(colhistogram, colhistogram).flatten()
    
    def getVector(self):
        if self.vector is not None:
            return self.vector
        else:
            #Berechnet alle drei Deskriptor Vektoren und fügt sie zusammen
            #hu moments
            hmv = cv2.HuMoments(cv2.moments(self.bwimg)).flatten()
            #haralick
            hav = mahotas.features.haralick(self.bwimg).mean(axis=0)
            #colorhistogram
            colhistogram = cv2.calcHist([cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)], [0,1,2], None, [Deskriptor.BINS, Deskriptor.BINS, Deskriptor.BINS], [0, 256, 0, 256, 0, 256])
            chv = cv2.normalize(colhistogram, colhistogram).flatten()
            return np.hstack([ hmv, hav, chv])
        
        
    @staticmethod   
    def getXY(src="D:/Developement/Datasets/Leaves/", save=False, name="" , pcaComponents=16):
        # Deskriptoren aus https://github.com/Gogul09/image-classification-python
        X = []
        y = []    
        for nr, folder in enumerate(os.listdir(src)):
            print("[+] processing ", folder)
            path = src + folder
            for imgpath in os.listdir(path):
                d = Deskriptor(path+"/"+imgpath)
                v = d.getVector()
                X.append(v)
                y.append(nr)        
        X = np.array(X)
        y = np.array(y)
        
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=pcaComponents))  #, GaussianNB()
        std_clf.fit(X, y)
        # Extract PCA from pipeline
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_std = pca_std.transform(scaler.transform(X))
        
        if save:
            np.save(name+"_X", X_std)
            np.save(name+"_y", y)


    @staticmethod
    def getXYSingle(src="D:/Developement/Datasets/Leaves/", save=False, name="" , pcaComponents=16):
        # für jedes bild einzeln & für alle Zusammen
        X = []
        y = []    
        for nr, folder in enumerate(os.listdir(src)):
            nr+=1 #klasse startet bei 1 nicht 0
            print("[+] processing ", folder)
            path = src + folder
            for imgpath in os.listdir(path):
                d = Deskriptor(path+"/"+imgpath)
                v = d.getVector()
                X.append(v)
                y.append(nr)        
        X = np.array(X)
        y = np.array(y)
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=16))  #, GaussianNB()
        std_clf.fit(X, y)
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_std = pca_std.transform(scaler.transform(X))
        if save:
            np.save(name+"_X", X_std)
            np.save(name+"_y", y)
                    
                
        for clas,folder in enumerate(os.listdir(src)):
            clas +=1
            print("[+] processing single ", folder)
            path = src + folder
            for nr, imgpath in enumerate(os.listdir(path)):
                d = Deskriptor(path+"/"+imgpath)
                os.rename(path+"/"+imgpath, path+"/"+str(nr)+".jpg")
                v = d.getVector()
                v = np.array(v)
                v = v.reshape((1, v.shape[0]))
                v = pca_std.transform(scaler.transform(v))
                y = np.array([clas])
                np.save(path+"/"+str(nr)+"_X", v)
                np.save(path+"/"+str(nr)+"_y", y)


#-----------------------------------------------------------------------------------------------------


class Des2(Deskriptor):
    def __init__(self, imgPath):
        img = cv2.imread(imgPath)
        self.img = cv2.resize(img, Deskriptor.IMAGESIZE)
        self.img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        self.bwimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.vector = None
    
    def getVector(self):
        if self.vector is not None:
            return self.vector
        else:
            #Berechnet alle drei Deskriptor Vektoren und fügt sie zusammen
            #hu moments
            hmv = cv2.HuMoments(cv2.moments(self.bwimg)).flatten()
            #haralick
            hav = mahotas.features.haralick(self.bwimg).mean(axis=0)
            #colorhistogram
            colhistogram = cv2.calcHist([cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)], [0,1,2], None, [Deskriptor.BINS, Deskriptor.BINS, Deskriptor.BINS], [0, 256, 0, 256, 0, 256])
            chv = cv2.normalize(colhistogram, colhistogram).flatten()

            # Initiate STAR detector
            star = cv2.FeatureDetector_create("STAR")       
            # Initiate BRIEF extractor
            brief = cv2.DescriptorExtractor_create("BRIEF")         
            # find the keypoints with STAR
            kp = star.detect(self.img,None)
            # compute the descriptors with BRIEF
            kp, des = brief.compute(img, kp)
            return np.hstack([ hmv, hav, chv, des])
        
if __name__ == "__main__":
    print("running")
    name = "withBrief"
    Des2.getXY(src="D:/Developement/Datasets/Leaves/", save=True, name=name , pcaComponents=16)
    
    #X = np.load(r"Data/Leaves/X_std_n16.npy")
    #y = np.load(r"Data/Leaves/y_n16.npy")
    
    X = np.load(name+"_X.npy")
    y = np.load(name+"_y.npy")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=1)
    svm = MultiSVM()
    svm.fit(X_train, y_train)
    
    print(svm.score(X_test, y_test))
    print(svm.detailedScore(X_test, y_test))
    
        
                

